"""Play a Go2 checkpoint on four fixed, five-obstacle lanes.

Run from this directory, as with play.py. Extra options: --difficulty, --episodes,
--max-steps, --cmd-vx, --collision-threshold, --output, and --mode.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import isaacgym  # noqa: F401; import before torch
from isaacgym import gymtorch, terrain_utils
import numpy as np
import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import task_registry
from legged_gym.envs.base import legged_robot as robot_module
from legged_gym.utils import get_args
from legged_gym.utils.helpers import class_to_dict, parse_sim_params, set_seed
from legged_gym.utils.terrain import Terrain


LANES = ("parkour", "parkour_hurdle", "parkour_step", "parkour_gap")
CLASS_IDS = (15, 16, 18, 19)
OBSTACLE_CENTERS = (3.2, 5.9, 8.6, 11.3, 14.0)
GOAL_X = (1.5, 4.1, 6.8, 9.5, 12.2, 14.9, 16.5)


def build_lane(terrain, name, difficulty):
    """Five obstacles, each followed by at least 1.6 m of level landing ground."""
    scale = terrain.horizontal_scale
    vertical = terrain.vertical_scale
    height = terrain.height_field_raw
    height[:] = 0
    width = height.shape[1]
    center_y = width // 2
    to_x = lambda meters: round(meters / scale)
    if name == "parkour_gap":
        gap_width = 0.1 + 0.7 * difficulty
        for center in OBSTACLE_CENTERS:
            a, b = to_x(center - gap_width / 2), to_x(center + gap_width / 2)
            height[a:b, :] = round(-0.7 / vertical)
    elif name == "parkour_hurdle":
        low, high = 0.1 + 0.1 * difficulty, 0.15 + 0.25 * difficulty
        hurdle_height = round((low + high) / (2 * vertical))
        for center in OBSTACLE_CENTERS:
            height[to_x(center - 0.15):to_x(center + 0.15), :] = hurdle_height
    elif name == "parkour_step":
        step_height = round((0.1 + 0.35 * difficulty) / vertical)
        for center in OBSTACLE_CENTERS:
            # Each raised platform has an up edge and a down edge.
            height[to_x(center - 0.55):to_x(center + 0.55), :] = step_height
    elif name == "parkour":
        incline = 0.25 * difficulty
        stone_width = to_x(1.0)
        for index, center in enumerate(OBSTACLE_CENTERS):
            a, b = to_x(center - 0.55), to_x(center + 0.55)
            height[a:b, :] = round(-0.7 / vertical)
            lateral = to_x(0.2 if index % 2 == 0 else -0.2)
            lo = center_y + lateral - stone_width // 2
            hi = lo + stone_width
            slope = np.linspace(-incline, incline, stone_width)
            if index % 2:
                slope = slope[::-1]
            height[a:b, lo:hi] = np.rint(slope / vertical).astype(height.dtype)[None, :]
    else:
        raise ValueError(f"Unknown lane: {name}")
    terrain.goals = np.array([[x, 2.0] for x in GOAL_X], dtype=np.float32)
    terrain.idx = CLASS_IDS[LANES.index(name)]
    return terrain


class FourLaneTerrain(Terrain):
    def curiculum(self, random=False, max_difficulty=False):
        if self.cfg.num_rows != 1 or self.cfg.num_cols != 4:
            raise ValueError("FourLaneTerrain requires one row and four columns")
        for column, name in enumerate(LANES):
            tile = terrain_utils.SubTerrain(
                "terrain", width=self.length_per_env_pixels,
                length=self.width_per_env_pixels,
                vertical_scale=self.cfg.vertical_scale,
                horizontal_scale=self.cfg.horizontal_scale)
            self.add_terrain_to_map(build_lane(tile, name, self.cfg.fixed_difficulty), 0, column)


class Go2LaneRobot(robot_module.LeggedRobot):
    def _update_terrain_curriculum(self, env_ids):
        return  # Keep each robot in its assigned lane after every reset.

    def _update_goals(self):
        # These lanes only progress in +X. A fast jump can pass a 0.2 m waypoint
        # without spending 0.1 s inside it; the standard timer then leaves the
        # target behind the robot and encourages a turn back toward it.
        for _ in range(self.cfg.terrain.num_goals):
            goal = self._gather_cur_goals()
            passed = ((self.cur_goal_idx < self.cfg.terrain.num_goals)
                      & (self.root_states[:, 0] >= goal[:, 0]))
            self.cur_goal_idx += passed.long()
        self.cur_goals = self._gather_cur_goals()
        self.next_goals = self._gather_cur_goals(future=1)
        self.reach_goal_timer[:] = 0
        self.reached_goal_ids = torch.norm(
            self.root_states[:, :2] - self.cur_goals[:, :2], dim=1
        ) < self.cfg.env.next_goal_threshold
        self.target_pos_rel = self.cur_goals[:, :2] - self.root_states[:, :2]
        self.next_target_pos_rel = self.next_goals[:, :2] - self.root_states[:, :2]
        self.target_yaw = torch.atan2(self.target_pos_rel[:, 1], self.target_pos_rel[:, 0])
        self.next_target_yaw = torch.atan2(self.next_target_pos_rel[:, 1], self.next_target_pos_rel[:, 0])

    def _reset_dofs(self, ids):
        self.dof_pos[ids] = self.default_dof_pos
        self.dof_vel[ids] = 0.
        ids = ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.dof_state), gymtorch.unwrap_tensor(ids), len(ids))

    def check_termination(self):
        super().check_termination()
        self.off_lane_buf = torch.abs(
            self.root_states[:, 1] - self.env_origins[:, 1]
        ) > (self.cfg.terrain.terrain_width / 2 - 0.2)
        self.reset_buf |= self.off_lane_buf

    def reset_idx(self, ids):
        if len(ids) and getattr(self, "record_terminal", False):
            for index in ids.tolist():
                self.terminal[index] = read_state(self, index)
        super().reset_idx(ids)


def read_state(env, index):
    force = env.contact_forces[index, env.termination_contact_indices]
    force = float(torch.norm(force, dim=-1).max()) if force.numel() else 0.
    roll, pitch = float(env.roll[index]), float(env.pitch[index])
    goal_index = int(env.cur_goal_idx[index])
    target = env.env_goals[index, min(goal_index, env.cfg.terrain.num_goals - 1)]
    return dict(goal_index=goal_index, roll=roll, pitch=pitch,
                base_height=float(env.root_states[index, 2]),
                timed_out=bool(env.time_out_buf[index]),
                command_x=float(env.commands[index, 0]),
                command_yaw_rate=float(env.commands[index, 2]),
                actual_x=float(env.base_lin_vel[index, 0]),
                world_vx=float(env.root_states[index, 7]),
                base_contact_force=force,
                position_x=float(env.root_states[index, 0]),
                position_y=float(env.root_states[index, 1]),
                target_x=float(target[0]), target_y=float(target[1]),
                yaw=float(env.yaw[index]), target_yaw=float(env.target_yaw[index]),
                off_lane=bool(env.off_lane_buf[index]))


def finish_episode(lane, episode_id, samples, num_goals, dt):
    last = samples[-1]
    fallen = abs(last["roll"]) > 1.5 or abs(last["pitch"]) > 1.5 or last["base_height"] < -0.25
    reason = ("fall" if fallen else "off_lane" if last["off_lane"]
              else "success" if last["goal_index"] >= num_goals
              else "timeout" if last["timed_out"] else "other")
    mean = lambda key: sum(row[key] for row in samples) / len(samples)
    return dict(lane=lane, episode_id=episode_id, reason=reason,
                success=int(reason == "success"), steps=len(samples),
                elapsed_s=len(samples) * dt,
                progress=min(1., last["goal_index"] / num_goals),
                collision_events=sum(row["collision_start"] for row in samples),
                collision_duration_s=sum(row["collision"] for row in samples) * dt,
                max_base_contact_force=max(row["base_contact_force"] for row in samples),
                mean_command_x=mean("command_x"), mean_actual_x=mean("actual_x"),
                mean_world_vx=mean("world_vx"),
                mean_abs_speed_error=mean("abs_speed_error"),
                mean_abs_roll=mean("abs_roll"), mean_abs_pitch=mean("abs_pitch"))


def write_csv(path, rows):
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_lane_goals(env):
    """Fail before play if a robot was assigned another lane's waypoints."""
    expected_types = torch.arange(len(LANES), device=env.device)
    if not torch.equal(env.terrain_types, expected_types):
        raise RuntimeError(f"Wrong lane assignment: {env.terrain_types.tolist()}")
    if not torch.all(env.terrain_levels == 0):
        raise RuntimeError(f"Wrong terrain levels: {env.terrain_levels.tolist()}")
    expected_x = torch.tensor(GOAL_X, device=env.device)
    if not torch.allclose(env.env_goals[:, :len(GOAL_X), 0], expected_x[None, :].expand(len(LANES), -1), atol=1e-4):
        raise RuntimeError("A lane has unexpected target X coordinates")
    expected_y = env.env_origins[:, 1]
    if not torch.allclose(env.env_goals[:, :len(GOAL_X), 1], expected_y[:, None].expand(-1, len(GOAL_X)), atol=1e-4):
        raise RuntimeError("A lane has another lane's target Y coordinates")
    return {lane: env.env_goals[index, :len(GOAL_X), :2].cpu().tolist()
            for index, lane in enumerate(LANES)}


def parse_args():
    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument("--difficulty", type=float, default=0.8)
    extra.add_argument("--episodes", type=int, default=20)
    extra.add_argument("--max-steps", type=int, default=3000)
    extra.add_argument("--cmd-vx", type=float, default=0.6)
    extra.add_argument("--collision-threshold", type=float, default=1.0)
    extra.add_argument("--mode", choices=("student", "teacher"), default="student")
    extra.add_argument("--output", default=str(Path(LEGGED_GYM_ROOT_DIR) / "evaluation" / "go2-four-lanes"))
    custom, remaining = extra.parse_known_args()
    original = sys.argv
    try:
        sys.argv = [original[0], *remaining]
        args = get_args()
    finally:
        sys.argv = original
    if not 0 <= custom.difficulty <= 1:
        extra.error("--difficulty must be between 0 and 1")
    if custom.episodes < 1 or custom.max_steps < 2 or custom.collision_threshold < 0:
        extra.error("episodes must be positive, max-steps at least 2, collision-threshold nonnegative")
    if not args.exptid:
        extra.error("--exptid is required")
    args.task = "go2"
    args.num_envs = 4
    args.use_camera = custom.mode == "student"
    return args, custom


def main():
    args, custom = parse_args()
    checkpoint_dir = Path(LEGGED_GYM_ROOT_DIR) / "logs" / args.proj_name / args.exptid
    if args.checkpoint < 0:
        files = sorted((path for path in checkpoint_dir.glob("model_*.pt")
                        if path.stem.split("_")[-1].isdigit()),
                       key=lambda path: int(path.stem.split("_")[-1]))
        if not files:
            raise FileNotFoundError(f"No model checkpoints in {checkpoint_dir}")
        args.checkpoint = int(files[-1].stem.split("_")[-1])
    checkpoint = checkpoint_dir / f"model_{args.checkpoint}.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if custom.mode == "student":
        saved = torch.load(checkpoint, map_location="cpu")
        required = ("model_state_dict", "depth_actor_state_dict", "depth_encoder_state_dict")
        missing = [key for key in required if key not in saved]
        if missing:
            raise KeyError(f"Student checkpoint lacks {missing}: {checkpoint}")
        del saved

    cfg, train_cfg = task_registry.get_cfgs("go2")
    cfg.env.num_envs = 4
    cfg.env.episode_length_s = custom.max_steps * cfg.sim.dt * cfg.control.decimation
    cfg.env.randomize_start_pos = False
    cfg.env.randomize_start_yaw = False
    cfg.env.randomize_start_pitch = False
    cfg.env.randomize_start_y = False
    cfg.terrain.num_rows = 1
    cfg.terrain.num_cols = 4
    cfg.terrain.num_goals = len(GOAL_X)
    cfg.terrain.max_init_terrain_level = 0
    cfg.terrain.curriculum = True  # FourLaneTerrain's fixed generator is called here.
    cfg.terrain.selected = False
    cfg.terrain.fixed_difficulty = custom.difficulty
    cfg.commands.curriculum = False
    cfg.commands.resampling_time = cfg.env.episode_length_s + 1
    cfg.commands.max_ranges.lin_vel_x = [custom.cmd_vx, custom.cmd_vx]
    cfg.commands.max_ranges.lin_vel_y = [0., 0.]
    cfg.commands.max_ranges.ang_vel_yaw = [0., 0.]
    cfg.commands.max_ranges.heading = [0., 0.]
    cfg.depth.use_camera = args.use_camera
    cfg.depth.angle = [0., 0.]
    cfg.depth.dis_noise = 0.
    cfg.noise.add_noise = False
    cfg.domain_rand.randomize_friction = True  # Observation requires this tensor.
    cfg.domain_rand.friction_range = [1., 1.]
    cfg.domain_rand.randomize_motor = False
    cfg.domain_rand.motor_strength_range = [1., 1.]
    cfg.domain_rand.randomize_base_mass = False
    cfg.domain_rand.randomize_base_com = False
    cfg.domain_rand.push_robots = False
    cfg.domain_rand.action_delay = True
    cfg.domain_rand.action_curr_step = [1]

    cfg.seed = args.seed if args.seed is not None else train_cfg.seed
    set_seed(cfg.seed)
    sim_params = parse_sim_params(args, {"sim": class_to_dict(cfg.sim)})
    original_terrain = robot_module.Terrain
    try:
        robot_module.Terrain = FourLaneTerrain
        env = Go2LaneRobot(cfg, sim_params, args.physics_engine, args.sim_device, args.headless)
    finally:
        robot_module.Terrain = original_terrain
    env.record_terminal = True
    env.terminal = {}
    env.episode_length_buf[:] = 0
    env.max_episode_length = custom.max_steps - 1
    env.max_episode_length_s = custom.max_steps * env.dt
    lane_goals = validate_lane_goals(env)
    for lane, goals in lane_goals.items():
        print(f"{lane}: start goal={goals[0]}, finish goal={goals[-1]}", flush=True)
    train_cfg.runner.resume = True
    train_cfg.runner.checkpoint = args.checkpoint
    runner, _, _ = task_registry.make_alg_runner(
        env=env, name="go2", args=args, train_cfg=train_cfg,
        log_root=str(checkpoint_dir), return_log_dir=True)
    if args.use_camera:
        if not runner.if_depth or runner.alg.depth_encoder is None or runner.alg.depth_actor is None:
            raise RuntimeError("Checkpoint runner has no student depth policy")
        encoder = runner.get_depth_encoder_inference_policy(device=env.device)
        actor = runner.get_depth_actor_inference_policy(device=env.device)
        latent = torch.zeros(env.num_envs, actor.scan_encoder_output_dim, device=env.device)
        yaw = torch.zeros(env.num_envs, 2, device=env.device)
        depth = env.depth_buffer[:, -1]
    else:
        encoder = None
        actor = runner.get_actor_critic(device=env.device).actor
        depth = None
    output = Path(custom.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "lane_goals.json").write_text(json.dumps(lane_goals, indent=2))
    obs = env.get_observations()
    counts = [0] * 4
    active = [[] for _ in LANES]
    previous_collision = [False] * 4
    episode_rows, step_rows = [], []
    with torch.no_grad():
        while min(counts) < custom.episodes:
            actor_obs = obs.clone()
            if encoder is not None:
                if depth is not None:
                    proprio = obs[:, :cfg.env.n_proprio].clone()
                    proprio[:, 6:8] = 0.
                    prediction = encoder(depth.clone(), proprio)
                    latent = prediction[:, :-2]
                    yaw = 1.5 * prediction[:, -2:]
                actor_obs[:, 6:8] = yaw
                actions = actor(actor_obs, hist_encoding=True, scandots_latent=latent)
            else:
                actions = actor(actor_obs, hist_encoding=True)
            obs, _, _, dones, info = env.step(actions)
            done_ids = set(dones.nonzero(as_tuple=False).flatten().tolist())
            for index, lane in enumerate(LANES):
                if counts[index] >= custom.episodes:
                    continue
                state = env.terminal[index] if index in done_ids else read_state(env, index)
                collision = state["base_contact_force"] > custom.collision_threshold
                row = dict(lane=lane, env_id=index, episode_id=counts[index],
                           step=len(active[index]) + 1,
                           timestamp_s=(len(active[index]) + 1) * env.dt,
                           goal_index=state["goal_index"],
                           position_x=state["position_x"], position_y=state["position_y"],
                           target_x=state["target_x"], target_y=state["target_y"],
                           command_x=state["command_x"], actual_x=state["actual_x"],
                           world_vx=state["world_vx"],
                           command_yaw_rate=state["command_yaw_rate"], yaw=state["yaw"],
                           target_yaw=state["target_yaw"],
                           actor_yaw_current=float(actor_obs[index, 6]),
                           actor_yaw_next=float(actor_obs[index, 7]),
                           abs_speed_error=abs(state["command_x"] - state["actual_x"]),
                           roll=state["roll"], pitch=state["pitch"],
                           base_height=state["base_height"], timed_out=int(state["timed_out"]),
                           abs_roll=abs(state["roll"]), abs_pitch=abs(state["pitch"]),
                           base_contact_force=state["base_contact_force"],
                           collision=int(collision),
                           collision_start=int(collision and not previous_collision[index]),
                           off_lane=int(state["off_lane"]),
                           depth_fresh=int(depth is not None))
                previous_collision[index] = collision
                active[index].append(row)
                step_rows.append(row)
                if index in done_ids:
                    episode_rows.append(finish_episode(lane, counts[index], active[index], cfg.terrain.num_goals, env.dt))
                    counts[index] += 1
                    active[index] = []
                    previous_collision[index] = False
                    print(f"{lane}: {counts[index]}/{custom.episodes} {episode_rows[-1]['reason']}", flush=True)
            if done_ids and encoder is not None:
                if encoder.hidden_states is not None:
                    encoder.hidden_states[:, list(done_ids), :] = 0.
                latent[list(done_ids)] = 0.
                yaw[list(done_ids)] = 0.
            env.terminal.clear()
            depth = info.get("depth") if encoder is not None else None
    write_csv(output / "episodes.csv", episode_rows)
    write_csv(output / "steps.csv", step_rows)
    summary = []
    for lane in LANES:
        rows = [row for row in episode_rows if row["lane"] == lane]
        successes = [row for row in rows if row["success"]]
        summary.append(dict(lane=lane, episodes=len(rows), successes=sum(row["success"] for row in rows),
                            success_rate=sum(row["success"] for row in rows) / len(rows),
                            fall_rate=sum(row["reason"] == "fall" for row in rows) / len(rows),
                            off_lane_rate=sum(row["reason"] == "off_lane" for row in rows) / len(rows),
                            timeout_rate=sum(row["reason"] == "timeout" for row in rows) / len(rows),
                            mean_progress=sum(row["progress"] for row in rows) / len(rows),
                            mean_success_time_s=(sum(row["elapsed_s"] for row in successes) / len(successes)
                                                 if successes else float("nan")),
                            mean_collision_events=sum(row["collision_events"] for row in rows) / len(rows),
                            mean_command_x=sum(row["mean_command_x"] for row in rows) / len(rows),
                            mean_actual_x=sum(row["mean_actual_x"] for row in rows) / len(rows),
                            mean_world_vx=sum(row["mean_world_vx"] for row in rows) / len(rows),
                            mean_abs_speed_error=sum(row["mean_abs_speed_error"] for row in rows) / len(rows)))
    write_csv(output / "summary.csv", summary)
    (output / "metadata.json").write_text(json.dumps(dict(
        checkpoint=str(checkpoint), mode=custom.mode, difficulty=custom.difficulty,
        episodes_per_lane=custom.episodes, max_steps=custom.max_steps,
        command_x=custom.cmd_vx, collision_threshold=custom.collision_threshold,
        lanes=LANES, obstacle_centers=OBSTACLE_CENTERS, goal_x=GOAL_X,
        goal_rule="advance when base X crosses each landing waypoint X",
        seed=cfg.seed, depth_update_interval=cfg.depth.update_interval), indent=2))
    print(f"Saved {output}")
    for row in summary:
        print(row)
    env.gym.destroy_sim(env.sim)


if __name__ == "__main__":
    main()
