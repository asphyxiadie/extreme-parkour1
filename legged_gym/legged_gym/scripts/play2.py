"""Record Go2 play evaluation, including a single-robot MoE gap test.

Examples (run from legged_gym/legged_gym/scripts):
  python play2.py --model teacher --teacher-run parkour_new/go2-attn-01 --teacher-checkpoint 14500
  python play2.py --model moe --moe-run parkour_new/go2-moe-01 --moe-checkpoint 14000
  python play2.py --model all --fixed-difficulty 0.8 --seed 1234
  python play2.py --model moe --gap-only --fixed-difficulty 0.8 --seed 1234
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import isaacgym  # noqa: F401 -- must precede torch
from isaacgym import gymapi, terrain_utils
import cv2
import numpy as np
import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import task_registry
from legged_gym.envs.base import legged_robot as robot_module
from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.utils import get_args
from legged_gym.utils.helpers import class_to_dict, parse_sim_params, set_seed
from legged_gym.utils.terrain import Terrain, parkour_gap_terrain


TERRAIN_NAMES = {15: 'parkour', 16: 'parkour_hurdle', 18: 'parkour_step',
                 19: 'parkour_gap', 20: 'demo'}
LANE_NAMES = ('parkour', 'parkour_hurdle', 'parkour_step', 'parkour_gap')
GAP_DEPTH_M = 0.5
STEP_FIELDS = ('model', 'env_id', 'terrain', 'terrain_level', 'episode_id', 'step',
               'time_s', 'goal_index', 'position_x', 'position_y', 'position_z',
               'target_x', 'target_y', 'final_goal_x', 'final_goal_y',
               'final_goal_distance_m', 'success_condition',
               'command_vx', 'actual_vx', 'world_vx',
               'roll_rad', 'pitch_rad', 'yaw_rad', 'ang_vel_x', 'ang_vel_y',
               'body_contact_force_n', 'collision', 'collision_start',
               'action_delta_abs_mean', 'depth_fresh', 'done', 'video_frame')


def snapshot(env, index):
    root = env.root_states[index]
    goal_index = int(env.cur_goal_idx[index].item())
    goal = env.env_goals[index, min(goal_index, env.cfg.terrain.num_goals - 1)]
    final_goal = env.env_goals[index, env.cfg.terrain.num_goals - 1]
    forces = env.contact_forces[index, env.termination_contact_indices]
    force = float(torch.linalg.vector_norm(forces, dim=-1).max().item()) if forces.numel() else 0.0
    terrain_class = int(env.env_class[index].item())
    return dict(
        terrain=TERRAIN_NAMES.get(terrain_class, f'class_{terrain_class}'),
        terrain_level=int(env.terrain_levels[index].item()),
        goal_index=goal_index,
        position_x=float(root[0].item()), position_y=float(root[1].item()),
        position_z=float(root[2].item()),
        target_x=float(goal[0].item()), target_y=float(goal[1].item()),
        final_goal_x=float(final_goal[0].item()), final_goal_y=float(final_goal[1].item()),
        final_goal_distance_m=float(torch.norm(root[:2] - final_goal[:2]).item()),
        success_condition=bool(env.play2_success_buf[index].item()),
        command_vx=float(env.commands[index, 0].item()),
        actual_vx=float(env.base_lin_vel[index, 0].item()),
        world_vx=float(root[7].item()),
        roll_rad=float(env.roll[index].item()), pitch_rad=float(env.pitch[index].item()),
        yaw_rad=float(env.yaw[index].item()),
        ang_vel_x=float(env.base_ang_vel[index, 0].item()),
        ang_vel_y=float(env.base_ang_vel[index, 1].item()),
        body_contact_force_n=force,
        timed_out=bool(env.time_out_buf[index].item()))


class Play2FixedGapTerrain(Terrain):
    def make_terrain(self, choice, difficulty):
        if not self.proportions[17] <= choice < self.proportions[18]:
            return super().make_terrain(choice, difficulty)

        tile = terrain_utils.SubTerrain(
            'terrain', width=self.length_per_env_pixels,
            length=self.width_per_env_pixels,
            vertical_scale=self.cfg.vertical_scale,
            horizontal_scale=self.cfg.horizontal_scale)
        parkour_gap_terrain(
            tile, num_gaps=self.num_goals - 2,
            gap_size=0.1 + 0.7 * difficulty,
            gap_depth=[GAP_DEPTH_M, GAP_DEPTH_M], pad_height=0,
            x_range=[0.8, 1.5], y_range=self.cfg.y_range,
            half_valid_width=[0.6, 1.2])
        floor_height = -round(GAP_DEPTH_M / self.cfg.vertical_scale)
        gap_cells = tile.height_field_raw == floor_height
        self.add_roughness(tile)
        tile.height_field_raw[gap_cells] = floor_height
        tile.idx = 19
        return tile


class RecordingRobot(LeggedRobot):
    def check_termination(self):
        super().check_termination()
        final_goal = self.env_goals[:, self.cfg.terrain.num_goals - 1, :2]
        final_distance = torch.norm(self.root_states[:, :2] - final_goal, dim=1)
        self.play2_success_buf = (
            (self.cur_goal_idx >= self.cfg.terrain.num_goals - 1)
            & (final_distance < self.cfg.env.next_goal_threshold))
        self.time_out_buf |= self.play2_success_buf
        self.reset_buf |= self.play2_success_buf

    def create_sim(self):
        # Headless video still needs a graphics context, including for the teacher.
        self.graphics_device_id = self.sim_device_id
        original_terrain = robot_module.Terrain
        try:
            robot_module.Terrain = Play2FixedGapTerrain
            super().create_sim()
        finally:
            robot_module.Terrain = original_terrain

    def _create_envs(self):
        super()._create_envs()
        self.record_cameras = []
        for index, handle in enumerate(self.envs):
            props = gymapi.CameraProperties()
            props.width = self.cfg.record_video_width
            props.height = self.cfg.record_video_height
            camera = self.gym.create_camera_sensor(handle, props)
            if camera < 0:
                raise RuntimeError(f'Cannot create recording camera for env {index}')
            self.record_cameras.append(camera)

    def reset_idx(self, env_ids):
        if len(env_ids) and getattr(self, 'record_terminal', False):
            self.terminal_states = {int(i): snapshot(self, int(i)) for i in env_ids.tolist()}
        super().reset_idx(env_ids)


class VideoRecorder:
    def __init__(self, env, output, width, height, stride, record_depth=False):
        self.env, self.output = env, output
        self.width, self.height, self.stride = width, height, stride
        self.record_depth = record_depth
        self.depth_width = width // 2 if record_depth else 0
        self.cameras = env.record_cameras
        self.writers = [None] * env.num_envs
        self.frames = [0] * env.num_envs
        self.latest_depth = [None] * env.num_envs
        self.fps = max(1, round(1 / (env.dt * stride)))
        (output / 'videos').mkdir(parents=True, exist_ok=True)

    def _open(self, index, episode_id, terrain):
        path = self.output / 'videos' / f'env{index}_{terrain}_episode{episode_id:02d}.mp4'
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'),
                                 self.fps, (self.width + self.depth_width, self.height))
        if not writer.isOpened():
            raise RuntimeError(f'Cannot open MP4 writer: {path}')
        self.writers[index] = writer
        self.frames[index] = 0
        return path

    def capture(self, counts, lengths, terrains, active, depth_input=None):
        """Capture the current state before the next control step can reset an episode."""
        if self.record_depth and depth_input is not None:
            images = depth_input.detach().cpu().numpy()
            for index in range(self.env.num_envs):
                if active[index]:
                    self.latest_depth[index] = images[index].copy()
        due = [i for i in range(self.env.num_envs)
               if active[i] and lengths[i] % self.stride == 0]
        if not due:
            return
        for index in due:
            root = self.env.root_states[index, :3].detach().cpu().numpy()
            # Match BaseTask.lookat(): its initial viewer follow offset is [0, 2, 1].
            camera_pos = gymapi.Vec3(float(root[0]), float(root[1]) + 2.0,
                                     float(root[2]) + 1.0)
            look_at = gymapi.Vec3(float(root[0]), float(root[1]), float(root[2]))
            self.env.gym.set_camera_location(self.cameras[index], self.env.envs[index],
                                             camera_pos, look_at)
        self.env.gym.step_graphics(self.env.sim)
        self.env.gym.render_all_camera_sensors(self.env.sim)
        for index in due:
            if self.writers[index] is None:
                self._open(index, counts[index], terrains[index])
            raw = self.env.gym.get_camera_image(self.env.sim, self.env.envs[index],
                                                self.cameras[index], gymapi.IMAGE_COLOR)
            rgba = np.asarray(raw).reshape(self.height, self.width, 4)
            frame = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
            if self.record_depth:
                panel = self.depth_panel(self.latest_depth[index])
                frame = np.concatenate((frame, panel), axis=1)
            self.writers[index].write(frame)
            self.frames[index] += 1

    def depth_panel(self, depth):
        if depth is None:
            gray = np.zeros((self.height, self.depth_width), dtype=np.uint8)
        else:
            # The depth encoder receives values in approximately [-0.5, 0.5].
            gray = np.clip((depth + 0.5) * 255, 0, 255).astype(np.uint8)
            gray = cv2.resize(gray, (self.depth_width, self.height),
                              interpolation=cv2.INTER_NEAREST)
        panel = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cv2.putText(panel, 'DEPTH INPUT', (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 2)
        return panel

    def close_episode(self, index):
        frames = self.frames[index]
        if self.writers[index] is not None:
            self.writers[index].release()
            self.writers[index] = None
        self.frames[index] = 0
        self.latest_depth[index] = None
        return frames

    def close(self):
        for index in range(len(self.writers)):
            self.close_episode(index)


def parse_args():
    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument('--model', choices=('teacher', 'moe', 'all'), default='all')
    extra.add_argument('--gap-only', action='store_true',
                       help='Evaluate only MoE on one parkour_gap robot')
    extra.add_argument('--teacher-run', default='parkour_new/go2-attn-01')
    extra.add_argument('--teacher-checkpoint', type=int, default=14500)
    extra.add_argument('--moe-run', default='parkour_new/go2-moe-01')
    extra.add_argument('--moe-checkpoint', type=int, default=14000)
    extra.add_argument('--episodes', type=int, default=20, help='Episodes per robot')
    extra.add_argument('--max-steps', type=int, default=3000)
    extra.add_argument('--collision-threshold', type=float, default=1.0)
    extra.add_argument('--video-width', type=int, default=640)
    extra.add_argument('--video-height', type=int, default=360)
    extra.add_argument('--video-stride', type=int, default=5,
                       help='Capture one frame every N control steps')
    extra.add_argument('--output', default='~/gpufree-data/parkour-data')
    opts, remaining = extra.parse_known_args()
    original = sys.argv
    try:
        sys.argv = [original[0], *remaining]
        args = get_args()
    finally:
        sys.argv = original
    if opts.episodes < 1 or opts.max_steps < 2 or opts.video_stride < 1:
        extra.error('episodes and video-stride must be positive; max-steps must be at least 2')
    if opts.video_width < 64 or opts.video_height < 64 or opts.collision_threshold < 0:
        extra.error('video dimensions must be at least 64; collision threshold cannot be negative')
    if args.fixed_difficulty is not None and not 0 <= args.fixed_difficulty <= 1:
        extra.error('--fixed-difficulty must be between 0 and 1')
    if args.graphics_device_id < 0:
        extra.error('Video recording requires a graphics device (graphics_device_id >= 0)')
    if opts.gap_only:
        if opts.model == 'teacher':
            extra.error('--gap-only requires --model moe')
        opts.model = 'moe'
        if opts.output == '~/gpufree-data/parkour-data':
            opts.output = '~/gpufree-data/parkour-data/moe-gap-0p5'
    args.task = 'go2'
    args.num_envs = 1 if opts.gap_only else 4
    args.headless = True  # Offscreen cameras still render through graphics_device_id.
    args.resume = False
    args.resumeid = None
    return opts, args


def checkpoint_path(run, number):
    directory = Path(run).expanduser()
    if not directory.is_absolute() and not directory.is_dir():
        directory = Path(LEGGED_GYM_ROOT_DIR) / 'logs' / run
    if number < 0:
        candidates = sorted((p for p in directory.glob('model_*.pt')
                             if p.stem.split('_')[-1].isdigit()),
                            key=lambda p: int(p.stem.split('_')[-1]))
        if not candidates:
            raise FileNotFoundError(f'No checkpoints in {directory}')
        return candidates[-1]
    path = directory / f'model_{number}.pt'
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_episode(model, env_id, episode_id, rows, last, dt, num_goals, frames):
    fallen = (abs(last['roll_rad']) > 1.5 or abs(last['pitch_rad']) > 1.5
              or last['position_z'] < -0.25)
    reason = ('fall' if fallen else 'success' if last['success_condition']
              else 'timeout' if last['timed_out'] else 'other')
    mean = lambda key: sum(r[key] for r in rows) / len(rows)
    action_deltas = [r['action_delta_abs_mean'] for r in rows
                     if r['action_delta_abs_mean'] != '']
    return dict(model=model, env_id=env_id, terrain=rows[0]['terrain'],
                terrain_level=rows[0]['terrain_level'], episode_id=episode_id,
                reason=reason, success=int(reason == 'success'),
                elapsed_s=len(rows) * dt, steps=len(rows),
                progress=(1.0 if reason == 'success' else min(1.0, last['goal_index'] / num_goals)),
                final_goal_distance_m=last['final_goal_distance_m'],
                mean_command_vx=mean('command_vx'), mean_actual_vx=mean('actual_vx'),
                mean_abs_speed_error=sum(abs(r['command_vx'] - r['actual_vx']) for r in rows) / len(rows),
                collision_events=sum(r['collision_start'] for r in rows),
                collision_rate=sum(r['collision'] for r in rows) / len(rows),
                collision_duration_s=sum(r['collision'] for r in rows) * dt,
                max_body_contact_force_n=max(r['body_contact_force_n'] for r in rows),
                mean_abs_roll_rad=sum(abs(r['roll_rad']) for r in rows) / len(rows),
                mean_abs_pitch_rad=sum(abs(r['pitch_rad']) for r in rows) / len(rows),
                rms_ang_vel_xy=math.sqrt(sum(r['ang_vel_x'] ** 2 + r['ang_vel_y'] ** 2
                                             for r in rows) / len(rows)),
                action_smoothness=(sum(action_deltas) / len(action_deltas)
                                   if action_deltas else float('nan')),
                video_frames=frames,
                video=f'videos/env{env_id}_{rows[0]["terrain"]}_episode{episode_id:02d}.mp4')


def make_cfg(args, opts, model):
    cfg, train_cfg = task_registry.get_cfgs('go2')
    cfg.env.num_envs = 1 if opts.gap_only else 4
    cfg.env.episode_length_s = opts.max_steps * cfg.sim.dt * cfg.control.decimation
    cfg.commands.resampling_time = cfg.env.episode_length_s + 1
    cfg.terrain.num_rows = 5
    cfg.terrain.num_cols = 1 if opts.gap_only else 5
    cfg.terrain.height = [0.02, 0.02]
    cfg.terrain.terrain_dict = dict.fromkeys(cfg.terrain.terrain_dict, 0.0)
    if opts.gap_only:
        cfg.terrain.terrain_dict['parkour_gap'] = 1.0
    else:
        cfg.terrain.terrain_dict.update({name: 0.2 for name in
                                         ('parkour', 'parkour_hurdle', 'parkour_step',
                                          'parkour_gap', 'demo')})
    cfg.terrain.terrain_proportions = list(cfg.terrain.terrain_dict.values())
    cfg.terrain.curriculum = False
    cfg.terrain.max_difficulty = True
    # play.py applies this when --use_camera is set. Use the same map for both
    # policies so teacher/MoE comparisons start from identical terrain tiles.
    cfg.terrain.y_range = [-0.1, 0.1]
    if args.fixed_difficulty is not None:
        cfg.terrain.fixed_difficulty = args.fixed_difficulty
    cfg.depth.use_camera = model == 'moe'
    cfg.record_video_width = opts.video_width
    cfg.record_video_height = opts.video_height
    cfg.depth.angle = [0, 1]
    cfg.depth.dis_noise = 0.0
    cfg.noise.add_noise = True
    cfg.domain_rand.randomize_friction = True
    cfg.domain_rand.push_robots = False
    cfg.domain_rand.randomize_base_mass = False
    cfg.domain_rand.randomize_base_com = False
    cfg.domain_rand.action_delay = args.delay
    if args.delay:
        cfg.domain_rand.action_curr_step = [1]
    cfg.seed = args.seed if args.seed is not None else train_cfg.seed
    train_cfg.depth_encoder.if_depth = model == 'moe'
    train_cfg.runner.resume = False
    return cfg, train_cfg


def run_model(model, checkpoint, opts, args, output):
    args.use_camera = model == 'moe'
    cfg, train_cfg = make_cfg(args, opts, model)
    set_seed(cfg.seed)
    params = parse_sim_params(args, {'sim': class_to_dict(cfg.sim)})
    env = RecordingRobot(cfg, params, args.physics_engine, args.sim_device, headless=True)
    env.record_terminal = True
    env.terminal_states = {}
    env.episode_length_buf[:] = 0
    env.max_episode_length = opts.max_steps - 1
    env.max_episode_length_s = opts.max_steps * env.dt
    classes = [TERRAIN_NAMES.get(int(x.item()), str(int(x.item()))) for x in env.env_class]
    expected = ('parkour_gap',) if opts.gap_only else LANE_NAMES
    if tuple(classes) != expected:
        raise RuntimeError(f'Expected terrain types {expected}, got {classes}')
    run_dir = output / model
    run_dir.mkdir(parents=True, exist_ok=True)
    recorder = None
    try:
        required = ('model_state_dict', 'estimator_state_dict')
        if model == 'moe':
            required += ('depth_actor_state_dict', 'depth_encoder_state_dict')
        saved = torch.load(checkpoint, map_location='cpu')
        missing = [name for name in required if name not in saved]
        del saved
        if missing:
            raise KeyError(f'{checkpoint} missing required model weights: {missing}')
        runner, _ = task_registry.make_alg_runner(
            env=env, name='go2', args=args, train_cfg=train_cfg,
            log_root=None)
        runner.load(str(checkpoint), load_optimizer=False)
        if model == 'moe':
            encoder = runner.get_depth_encoder_inference_policy(device=env.device)
            actor = runner.get_depth_actor_inference_policy(device=env.device)
            latent = torch.zeros(env.num_envs, actor.scan_encoder_output_dim, device=env.device)
            yaw = torch.zeros(env.num_envs, 2, device=env.device)
            depth = env.depth_buffer[:, -1]
        else:
            encoder = None
            actor = runner.get_actor_critic(device=env.device).actor
            depth = None
        recorder = VideoRecorder(env, run_dir, opts.video_width, opts.video_height,
                                 opts.video_stride, record_depth=model == 'moe')
        obs = env.get_observations()
        counts = [0] * env.num_envs
        lengths = [0] * env.num_envs
        active = [[] for _ in range(env.num_envs)]
        previous_actions = [None] * env.num_envs
        previous_collision = [False] * env.num_envs
        episodes = []
        with (run_dir / 'steps.csv').open('w', newline='') as file, torch.no_grad():
            writer = csv.DictWriter(file, fieldnames=STEP_FIELDS)
            writer.writeheader()
            while min(counts) < opts.episodes:
                recording = [n < opts.episodes for n in counts]
                recorder.capture(counts, lengths, classes, recording, depth_input=depth)
                actor_obs = obs.clone()
                if encoder is not None:
                    if depth is not None:
                        proprio = actor_obs[:, :cfg.env.n_proprio].clone()
                        proprio[:, 6:8] = 0.0
                        prediction = encoder(depth.clone(), proprio)
                        latent = prediction[:, :-2]
                        yaw = 1.5 * prediction[:, -2:]
                    actor_obs[:, 6:8] = yaw
                    actions = actor(actor_obs, hist_encoding=True, scandots_latent=latent)
                else:
                    actions = actor(actor_obs, hist_encoding=True)
                obs, _, _, dones, info = env.step(actions.detach())
                done_ids = set(dones.nonzero(as_tuple=False).flatten().tolist())
                for index in range(env.num_envs):
                    if counts[index] >= opts.episodes:
                        continue
                    state = (env.terminal_states[index] if index in done_ids
                             else snapshot(env, index))
                    if state['terrain'] != classes[index]:
                        raise RuntimeError(f'Env {index} moved to another terrain')
                    collision = state['body_contact_force_n'] > opts.collision_threshold
                    action = actions[index].detach().cpu().numpy()
                    delta = (float(np.abs(action - previous_actions[index]).mean())
                             if previous_actions[index] is not None else '')
                    row = {key: state[key] for key in STEP_FIELDS if key in state}
                    row.update(model=model, env_id=index, episode_id=counts[index],
                               step=lengths[index] + 1, time_s=(lengths[index] + 1) * env.dt,
                               collision=int(collision),
                               collision_start=int(collision and not previous_collision[index]),
                               action_delta_abs_mean=delta, depth_fresh=int(depth is not None),
                               done=int(index in done_ids),
                               video_frame=int(lengths[index] % opts.video_stride == 0))
                    writer.writerow(row)
                    active[index].append(dict(row, timed_out=state['timed_out']))
                    lengths[index] += 1
                    previous_actions[index] = action.copy()
                    previous_collision[index] = collision
                    if index in done_ids:
                        frames = recorder.close_episode(index)
                        ep = summarize_episode(model, index, counts[index], active[index],
                                               state, env.dt, cfg.terrain.num_goals, frames)
                        episodes.append(ep)
                        counts[index] += 1
                        lengths[index] = 0
                        active[index] = []
                        previous_actions[index] = None
                        previous_collision[index] = False
                        print(f'{model}: {classes[index]} {counts[index]}/{opts.episodes} '
                              f'{ep["reason"]}', flush=True)
                if done_ids and encoder is not None:
                    if encoder.hidden_states is not None:
                        encoder.hidden_states[:, list(done_ids), :] = 0.0
                    latent[list(done_ids)] = 0.0
                    yaw[list(done_ids)] = 0.0
                env.terminal_states.clear()
                depth = info.get('depth') if encoder is not None else None
        write_csv(run_dir / 'episodes.csv', episodes)
        summary = []
        for index, terrain in enumerate(classes):
            rows = [r for r in episodes if r['env_id'] == index]
            successes = [r for r in rows if r['success']]
            summary.append(dict(model=model, env_id=index, terrain=terrain,
                                episodes=len(rows), successes=len(successes),
                                success_rate=len(successes) / len(rows),
                                fall_rate=sum(r['reason'] == 'fall' for r in rows) / len(rows),
                                mean_success_time_s=(sum(r['elapsed_s'] for r in successes) / len(successes)
                                                     if successes else float('nan')),
                                mean_collision_rate=sum(r['collision_rate'] for r in rows) / len(rows),
                                mean_collision_events=sum(r['collision_events'] for r in rows) / len(rows),
                                mean_actual_vx=sum(r['mean_actual_vx'] for r in rows) / len(rows),
                                mean_abs_speed_error=sum(r['mean_abs_speed_error'] for r in rows) / len(rows),
                                mean_abs_roll_rad=sum(r['mean_abs_roll_rad'] for r in rows) / len(rows),
                                mean_abs_pitch_rad=sum(r['mean_abs_pitch_rad'] for r in rows) / len(rows),
                                mean_rms_ang_vel_xy=sum(r['rms_ang_vel_xy'] for r in rows) / len(rows),
                                mean_action_smoothness=sum(r['action_smoothness'] for r in rows) / len(rows)))
        write_csv(run_dir / 'summary.csv', summary)
        (run_dir / 'metadata.json').write_text(json.dumps(dict(
            model=model, checkpoint=str(checkpoint), seed=cfg.seed,
            gap_only=opts.gap_only, num_envs=env.num_envs,
            episodes_per_robot=opts.episodes, max_steps=opts.max_steps,
            terrain_names=classes, terrain_levels=[int(x.item()) for x in env.terrain_levels],
            terrain_y_range_m=cfg.terrain.y_range,
            parkour_gap_depth_m=GAP_DEPTH_M,
            fixed_difficulty=args.fixed_difficulty,
            default_difficulty_range=[0.7, 1.0] if args.fixed_difficulty is None else None,
            collision_threshold_n=opts.collision_threshold, control_dt_s=env.dt,
            action_delay_enabled=cfg.domain_rand.action_delay,
            depth_camera_enabled=cfg.depth.use_camera,
            video_fps=recorder.fps, video_stride=opts.video_stride,
            video_camera_follow_offset_m=[0.0, 2.0, 1.0],
            video_camera_fov='Isaac Gym CameraProperties default, as in the play.py viewer',
            video_layout=('third-person left + policy depth input right' if encoder is not None
                          else 'third-person only'),
            video_note='Frames show the state before each sampled control step; the terminal state is in steps.csv.',
            depth_video_note=('The right panel holds the most recent depth input between camera updates; '
                              'it is blank after reset until the next input.' if encoder is not None else None),
            success_rule=('first 7 goals reached, then base XY within next_goal_threshold '
                          'of goal 8; fall takes precedence'),
            final_goal_threshold_m=cfg.env.next_goal_threshold,
            action_smoothness='mean absolute change in the 12 policy actions per control step',
            depth_update_interval=cfg.depth.update_interval if encoder is not None else None,
        ), indent=2))
        print(f'Saved {run_dir}', flush=True)
    finally:
        if recorder is not None:
            recorder.close()
        env.gym.destroy_sim(env.sim)


def main():
    opts, args = parse_args()
    output = Path(opts.output).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    selections = []
    if opts.model in ('teacher', 'all'):
        selections.append(('teacher', checkpoint_path(opts.teacher_run, opts.teacher_checkpoint)))
    if opts.model in ('moe', 'all'):
        selections.append(('moe', checkpoint_path(opts.moe_run, opts.moe_checkpoint)))
    for model, checkpoint in selections:
        print(f'Evaluating {model}: {checkpoint}', flush=True)
        run_model(model, checkpoint, opts, args, output)
    comparison = []
    for model, _ in selections:
        with (output / model / 'summary.csv').open(newline='') as file:
            comparison.extend(csv.DictReader(file))
    write_csv(output / 'comparison.csv', comparison)


if __name__ == '__main__':
    main()
