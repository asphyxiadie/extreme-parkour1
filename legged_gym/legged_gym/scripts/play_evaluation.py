"""Matched Go2 play evaluation; run with Isaac Gym installed on a GPU host."""

import argparse
import json
from copy import deepcopy
from pathlib import Path

import isaacgym  # noqa: F401; Isaac Gym must be imported before torch.
from isaacgym import gymapi, gymtorch, gymutil
import numpy as np
import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import task_registry
from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.utils.helpers import class_to_dict, set_seed
from legged_gym.utils.evaluation_metrics import (EpisodeAccumulator, plot_results,
                                                   summarize, termination_reason, write_csv)
from rsl_rl.modules.actor_critic import ActorCriticRMA
from rsl_rl.modules.depth_backbone import DepthOnlyFCBackbone58x87, RecurrentDepthBackbone
from rsl_rl.modules.estimator import Estimator


MODELS = ("teacher", "student", "moe")


class LegacyDepthEncoder(RecurrentDepthBackbone):
    """Exact pre-MoE CNN + GRU inference path for strict old-student loading."""

    def __init__(self, backbone, cfg):
        super().__init__(backbone, cfg)
        del self.terrain_moe

    def forward(self, depth_image, proprioception):
        depth_latent = self.base_backbone(depth_image)
        depth_latent = self.combination_mlp(torch.cat((depth_latent, proprioception), dim=-1))
        output, self.hidden_states = self.rnn(depth_latent[:, None, :], self.hidden_states)
        return self.output_mlp(output.squeeze(1))


def clear_gru_state(encoder, done):
    if encoder is not None and encoder.hidden_states is not None:
        encoder.hidden_states[:, done, :] = 0


def attention_weights(module, scan, proprioception):
    """Separate diagnostic pass; never changes the actor's action calculation."""
    tokens = module.scan_projection(scan.reshape(scan.shape[0], -1, module.patch_size)) + module.scan_position
    query = module.proprio_projection(proprioception).unsqueeze(1)
    # PyTorch 1.10 does not expose per-head weights from MultiheadAttention.
    # Reconstruct QK softmax using its *same* projection weights, without touching the actor pass.
    attention = module.attention
    dim = attention.embed_dim
    heads = attention.num_heads
    q_weight, k_weight, _ = attention.in_proj_weight.split(dim, dim=0)
    q_bias, k_bias, _ = attention.in_proj_bias.split(dim, dim=0)
    q = torch.nn.functional.linear(query, q_weight, q_bias).reshape(query.shape[0], -1, heads, dim // heads).transpose(1, 2)
    k = torch.nn.functional.linear(tokens, k_weight, k_bias).reshape(tokens.shape[0], -1, heads, dim // heads).transpose(1, 2)
    weights = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / (dim // heads) ** .5, dim=-1)
    return weights.detach().cpu().numpy()  # [env, head, query=1, token=12]


def snapshot(env, ids):
    """Called inside reset_idx before the environment overwrites terminal state."""
    result = {}
    for index in ids.tolist():
        copy = lambda tensor: tensor[index].detach().cpu().numpy().copy()
        base_forces = env.contact_forces[index, env.termination_contact_indices]
        result[index] = dict(position=copy(env.root_states[:, :3]),
                             quaternion=copy(env.root_states[:, 3:7]),
                             world_linear_velocity=copy(env.root_states[:, 7:10]),
                             linear_velocity=copy(env.base_lin_vel), angular_velocity=copy(env.base_ang_vel),
                             joint_position=copy(env.dof_pos), joint_velocity=copy(env.dof_vel),
                             torque=copy(env.torques), goal=copy(env.cur_goals), command=copy(env.commands),
                             goal_index=int(env.cur_goal_idx[index]),
                             roll=float(env.roll[index]), pitch=float(env.pitch[index]), yaw=float(env.yaw[index]),
                             target_yaw=float(env.target_yaw[index]),
                             base_contact_force=float(torch.norm(base_forces, dim=-1).max()) if base_forces.numel() else 0.,
                             timed_out=bool(env.time_out_buf[index]),
                             fallen=bool(abs(env.roll[index]) > 1.5 or abs(env.pitch[index]) > 1.5 or
                                         env.root_states[index, 2] < -0.25))
    return result


class EvaluationRobot(LeggedRobot):
    """Evaluation-only deterministic joint reset and pre-reset state hook."""

    def _reset_dofs(self, ids):
        self.dof_pos[ids] = self.default_dof_pos
        self.dof_vel[ids] = 0.
        handles = ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(handles), len(handles))

    def reset_idx(self, ids):
        if len(ids) and getattr(self, "capture_resets", False):
            self.terminal_snapshots.update(snapshot(self, ids))
        super().reset_idx(ids)


def configure_env(cfg, terrain, num_envs, camera, max_steps):
    cfg.env.num_envs = num_envs
    cfg.env.episode_length_s = max_steps * cfg.sim.dt * cfg.control.decimation
    cfg.terrain.num_rows = 10
    cfg.terrain.num_cols = num_envs
    cfg.terrain.curriculum = True  # Generate fixed row difficulty i / 9.
    cfg.terrain.selected = False
    cfg.terrain.terrain_dict = {key: float(key == terrain) for key in cfg.terrain.terrain_dict}
    cfg.terrain.terrain_proportions = list(cfg.terrain.terrain_dict.values())
    cfg.commands.curriculum = False
    cfg.commands.resampling_time = cfg.env.episode_length_s + 1
    cfg.commands.max_ranges.lin_vel_x = [0.6, 0.6]
    cfg.commands.max_ranges.lin_vel_y = [0., 0.]
    cfg.commands.max_ranges.ang_vel_yaw = [0., 0.]
    cfg.commands.max_ranges.heading = [0., 0.]
    cfg.noise.add_noise = False
    # The observation builder always reads friction_coeffs_tensor. Keep its
    # initialization path active, with a degenerate range for deterministic friction.
    cfg.domain_rand.randomize_friction = True
    cfg.domain_rand.friction_range = [1., 1.]
    cfg.domain_rand.randomize_base_mass = False
    cfg.domain_rand.randomize_base_com = False
    cfg.domain_rand.randomize_motor = False
    cfg.domain_rand.motor_strength_range = [1., 1.]
    cfg.domain_rand.push_robots = False
    cfg.domain_rand.action_delay = True
    cfg.domain_rand.action_curr_step = [1]
    cfg.env.randomize_start_pos = False
    cfg.env.randomize_start_yaw = False
    cfg.env.randomize_start_pitch = False
    cfg.env.randomize_start_y = False
    cfg.depth.use_camera = camera
    cfg.depth.dis_noise = 0.
    return cfg


def create_env(cfg, device, level, seed, max_steps):
    set_seed(seed)
    params = gymapi.SimParams()
    gymutil.parse_sim_config(class_to_dict(cfg.sim), params)
    params.physx.use_gpu = device.startswith("cuda")
    params.use_gpu_pipeline = device.startswith("cuda")
    env = EvaluationRobot(cfg, params, gymapi.SIM_PHYSX, device, headless=True)
    env.cfg.terrain.curriculum = False  # Freeze row after terrain generation.
    env.terrain_levels[:] = level
    env.env_origins[:] = env.terrain_origins[level, env.terrain_types]
    env.env_class[:] = env.terrain_class[level, env.terrain_types]
    goals = env.terrain_goals[level, env.terrain_types]
    env.env_goals[:] = torch.cat((goals, goals[:, -1:].repeat(1, cfg.env.num_future_goal_obs, 1)), dim=1)
    env.cur_goals = env._gather_cur_goals()
    env.next_goals = env._gather_cur_goals(future=1)
    env.reset_idx(torch.arange(env.num_envs, device=env.device))
    env.post_physics_step()
    env.episode_length_buf[:] = 0
    env.max_episode_length = max_steps - 1  # check_termination uses >.
    env.max_episode_length_s = max_steps * env.dt
    env.capture_resets = True
    env.terminal_snapshots = {}
    return env


def load_policy(model, checkpoint, env, cfg, device):
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    data = torch.load(checkpoint, map_location=device)
    _, train_cfg = task_registry.get_cfgs("go2")
    policy = ActorCriticRMA(cfg.env.n_proprio, cfg.env.n_scan, env.num_obs,
                            cfg.env.n_priv_latent, cfg.env.n_priv, cfg.env.history_len,
                            env.num_actions, **class_to_dict(train_cfg.policy)).to(device)
    policy.load_state_dict(data["model_state_dict"], strict=True)
    policy.eval()
    estimator = Estimator(cfg.env.n_proprio, cfg.env.n_priv, train_cfg.estimator.hidden_dims).to(device)
    estimator.load_state_dict(data["estimator_state_dict"], strict=True)
    if model == "teacher":
        return policy.actor, None
    if "depth_encoder_state_dict" not in data or "depth_actor_state_dict" not in data:
        raise KeyError(f"{checkpoint} is missing student depth weights")
    actor = deepcopy(policy.actor)
    actor.load_state_dict(data["depth_actor_state_dict"], strict=True)
    actor.eval()
    backbone = DepthOnlyFCBackbone58x87(cfg.env.n_proprio, actor.scan_encoder_output_dim, 512)
    encoder_class = LegacyDepthEncoder if model == "student" else RecurrentDepthBackbone
    encoder = encoder_class(backbone, cfg).to(device)
    encoder.load_state_dict(data["depth_encoder_state_dict"], strict=True)
    encoder.eval()
    return actor, encoder


def moe_diagnostics(encoder, raw_latent):
    moe = encoder.terrain_moe
    gate = moe.last_gate_weights
    outputs = torch.stack([expert(raw_latent) for expert in moe.experts], dim=1)
    return [dict(gate_weights=gate[i].cpu().numpy(),
                 expert_l2=torch.norm(outputs[i], dim=-1).cpu().numpy(),
                 weighted_l2=torch.norm(gate[i, :, None] * outputs[i], dim=-1).cpu().numpy(),
                 residual_scale=float(moe.residual_scale),
                 router_entropy=float(-(gate[i] * torch.log(gate[i] + 1e-12)).sum()),
                 top1=int(gate[i].argmax())) for i in range(raw_latent.shape[0])]


def run_one(args, model, terrain, level, checkpoint):
    cfg, _ = task_registry.get_cfgs("go2")
    cfg = configure_env(deepcopy(cfg), terrain, min(args.num_envs, args.episodes), model != "teacher", args.max_steps)
    env = create_env(cfg, args.device, level, args.seed, args.max_steps)
    actor, encoder = load_policy(model, checkpoint, env, cfg, args.device)
    raw_latent = {}
    if model == "moe" and args.record_moe:
        encoder.terrain_moe.register_forward_pre_hook(lambda _, inputs: raw_latent.update(value=inputs[0].detach()))
    obs = env.get_observations()
    depth = env.depth_buffer[:, -1] if encoder is not None else None
    latent = torch.zeros(env.num_envs, actor.scan_encoder_output_dim, device=env.device)
    yaw = torch.zeros(env.num_envs, 2, device=env.device)
    active = {}
    counts = [0] * env.num_envs
    episodes = []
    retained_steps = []
    for env_id in range(env.num_envs):
        active[env_id] = EpisodeAccumulator(model, terrain, level, int(env.env_class[env_id]), env_id, counts[env_id], env.dt,
                                            cfg.terrain.num_goals, args.collision_threshold)
    with torch.no_grad():
        for _ in range(args.episodes * args.max_steps + args.max_steps):
            fresh = depth is not None and encoder is not None
            moe_rows = None
            if fresh:
                prop = obs[:, :cfg.env.n_proprio].clone()
                prop[:, 6:8] = 0
                output = encoder(depth.clone(), prop)
                latent = output[:, :-2]
                yaw = 1.5 * output[:, -2:]
                if model == "moe" and args.record_moe:
                    moe_rows = moe_diagnostics(encoder, raw_latent["value"])
            obs_actor = obs.clone()
            if encoder is not None:
                valid = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
                obs_actor[valid, 6:8] = yaw[valid]
            if model == "teacher":
                action = actor(obs_actor, hist_encoding=True)
                attn = (attention_weights(actor.scan_encoder, obs_actor[:, cfg.env.n_proprio:cfg.env.n_proprio + cfg.env.n_scan],
                                          obs_actor[:, :cfg.env.n_proprio]) if args.record_attention else None)
            else:
                action = actor(obs_actor, hist_encoding=True, scandots_latent=latent)
                attn = None
            goal_before_step = env.cur_goals.detach().cpu().numpy().copy()
            next_obs, _, _, dones, infos = env.step(action)
            done_ids = set(dones.nonzero(as_tuple=False).flatten().tolist())
            current = snapshot(env, torch.arange(env.num_envs, device=env.device))
            for env_id, accumulator in active.items():
                state = env.terminal_snapshots[env_id] if env_id in done_ids else current[env_id]
                state["goal"] = goal_before_step[env_id]
                accumulator.add(state, action[env_id].cpu().numpy(), fresh,
                                moe_rows[env_id] if moe_rows is not None else None,
                                attn[env_id] if attn is not None else None)
            for env_id in sorted(done_ids):
                accumulator = active[env_id]
                state = env.terminal_snapshots[env_id]
                reason = termination_reason(state["goal_index"], cfg.terrain.num_goals,
                                            state["fallen"], state["timed_out"], True)
                if len(episodes) < args.episodes:
                    episodes.append(accumulator.finish(reason))
                    if args.record_steps:
                        retained_steps.extend(accumulator.rows)
                    elif args.record_moe or args.record_attention:
                        retained_steps.extend(row for row in accumulator.rows if row["moe"] is not None or row["attention"] is not None)
                counts[env_id] += 1
                active[env_id] = EpisodeAccumulator(model, terrain, level, int(env.env_class[env_id]), env_id, counts[env_id], env.dt,
                                                    cfg.terrain.num_goals, args.collision_threshold)
            env.terminal_snapshots.clear()
            if done_ids and encoder is not None:
                mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
                mask[list(done_ids)] = True
                clear_gru_state(encoder, mask)
                latent[mask] = 0
                yaw[mask] = 0
            obs = next_obs
            depth = infos.get("depth")
            if len(episodes) >= args.episodes:
                env.gym.destroy_sim(env.sim)
                return episodes, retained_steps
    raise RuntimeError("episode budget exhausted before collecting requested completions")


def save_npz(path, episodes, steps):
    arrays = {"episode_model": np.array([e["model"] for e in episodes]),
              "episode_terrain": np.array([e["terrain"] for e in episodes]),
              "episode_level": np.array([e["level"] for e in episodes], dtype=np.int16),
              "episode_env_id": np.array([e["env_id"] for e in episodes], dtype=np.int32),
              "episode_id": np.array([e["episode_id"] for e in episodes], dtype=np.int32),
              "episode_success": np.array([e["success"] for e in episodes], dtype=np.int8),
              "step_model": np.array([s["model"] for s in steps]),
              "step_terrain": np.array([s["terrain"] for s in steps]),
              "step_level": np.array([s["level"] for s in steps], dtype=np.int16),
              "step_env_id": np.array([s["env_id"] for s in steps], dtype=np.int32),
              "step_episode_id": np.array([s["episode_id"] for s in steps], dtype=np.int32),
              "step_timestamp_s": np.array([s["timestamp_s"] for s in steps], dtype=np.float32),
              "step_collision": np.array([s["collision"] for s in steps], dtype=np.int8),
              "step_base_contact_force": np.array([s["base_contact_force"] for s in steps], dtype=np.float32),
              "step_goal_index": np.array([s["goal_index"] for s in steps], dtype=np.int16),
              "step_depth_fresh": np.array([s["depth_fresh"] for s in steps], dtype=np.int8)}
    for key in ("position", "quaternion", "linear_velocity", "angular_velocity", "joint_position",
                "joint_velocity", "torque", "action", "goal", "command"):
        if steps:
            arrays[key] = np.stack([row[key] for row in steps]).astype(np.float32)
    moe_indices = [i for i, row in enumerate(steps) if row["moe"] is not None]
    if moe_indices:
        arrays["moe_step_index"] = np.asarray(moe_indices, dtype=np.int32)
        for key in ("gate_weights", "expert_l2", "weighted_l2"):
            arrays["moe_" + key] = np.stack([steps[i]["moe"][key] for i in moe_indices]).astype(np.float32)
        for key in ("residual_scale", "router_entropy", "top1"):
            arrays["moe_" + key] = np.asarray([steps[i]["moe"][key] for i in moe_indices])
    attention_indices = [i for i, row in enumerate(steps) if row["attention"] is not None]
    if attention_indices:
        arrays["attention_step_index"] = np.asarray(attention_indices, dtype=np.int32)
        arrays["attention_weights"] = np.stack([steps[i]["attention"] for i in attention_indices]).astype(np.float32)
    np.savez_compressed(path, **arrays)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=(*MODELS, "all"), default="all")
    log_root = Path(LEGGED_GYM_ROOT_DIR) / "logs"
    default_logs = log_root / "parkour_new" if (log_root / "parkour_new" / "go2-attn-01").is_dir() else log_root
    parser.add_argument("--teacher-run", default=str(default_logs / "go2-attn-01"))
    parser.add_argument("--student-run", default=str(default_logs / "go2-vision-01"))
    parser.add_argument("--moe-run", default=str(default_logs / "go2-moe-01"))
    parser.add_argument("--checkpoint", type=int, help="Override checkpoint for one selected model")
    parser.add_argument("--teacher-checkpoint", type=int, default=14500)
    parser.add_argument("--student-checkpoint", type=int, default=9500)
    parser.add_argument("--moe-checkpoint", type=int, default=14000)
    parser.add_argument("--terrain", default="all")
    parser.add_argument("--level", default="all", help="Terrain row 0..9, or all")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--episodes", type=int, default=20, help="Completed episodes per model/terrain/level")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--record-steps", action="store_true")
    parser.add_argument("--record-moe", action="store_true")
    parser.add_argument("--record-attention", action="store_true")
    parser.add_argument("--collision-threshold", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default=str(Path(LEGGED_GYM_ROOT_DIR) / "evaluation"))
    args = parser.parse_args()
    if args.episodes < 1 or args.num_envs < 1 or args.max_steps < 2:
        parser.error("episodes and num-envs must be positive; max-steps must be at least 2")
    if args.checkpoint is not None and args.model == "all":
        parser.error("--checkpoint requires a single --model")
    terrain_names = list(task_registry.get_cfgs("go2")[0].terrain.terrain_dict)
    if args.terrain != "all" and args.terrain not in terrain_names:
        parser.error("unknown terrain; choose one of: " + ", ".join(terrain_names))
    terrains = [name for name in terrain_names if name.startswith("parkour")] if args.terrain == "all" else [args.terrain]
    levels = list(range(10)) if args.level == "all" else [int(args.level)]
    if any(level not in range(10) for level in levels):
        parser.error("level must be 0..9")
    models = MODELS if args.model == "all" else (args.model,)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    episodes, steps = [], []
    for terrain in terrains:
        for level in levels:
            for model in models:
                number = args.checkpoint if args.checkpoint is not None else getattr(args, model + "_checkpoint")
                checkpoint = Path(getattr(args, model + "_run")) / f"model_{number}.pt"
                print(f"Evaluating {model}, {terrain}, level {level}: {checkpoint}", flush=True)
                episode_rows, step_rows = run_one(args, model, terrain, level, checkpoint)
                episodes.extend(episode_rows)
                steps.extend(step_rows)
                write_csv(output / "episodes.csv", episodes)
                write_csv(output / "summary.csv", summarize(episodes))
                if args.record_steps or args.record_moe or args.record_attention:
                    write_csv(output / "steps.csv", steps)
                save_npz(output / "records.npz", episodes, steps)
    plot_results(output, summarize(episodes), steps)
    (output / "metadata.json").write_text(json.dumps(dict(seed=args.seed, models=models, terrains=terrains,
        levels=levels, episodes=args.episodes, max_steps=args.max_steps, collision_threshold=args.collision_threshold,
        checkpoints={model: str(Path(getattr(args, model + "_run")) /
                                f'model_{args.checkpoint if args.checkpoint is not None else getattr(args, model + "_checkpoint")}.pt')
                     for model in models},
        depth_update_interval=5, scandot_tokens=12, scandot_patch_size=11), indent=2))


if __name__ == "__main__":
    main()
