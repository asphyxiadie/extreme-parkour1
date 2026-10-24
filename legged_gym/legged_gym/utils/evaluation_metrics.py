"""Policy-independent episode metrics and evaluation artifact writers."""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def termination_reason(goal_index, num_goals, fallen, timed_out, done):
    if not done:
        return "running"
    if fallen:
        return "fall"
    if goal_index >= num_goals:
        return "success"
    if timed_out:
        return "timeout"
    return "other"


class EpisodeAccumulator:
    """Collect control-step measurements; all inputs are plain scalars or arrays."""

    def __init__(self, model, terrain, level, terrain_class_id, env_id, episode_id, dt, num_goals, collision_threshold):
        self.meta = dict(model=model, terrain=terrain, level=level, terrain_class_id=terrain_class_id,
                         env_id=env_id, episode_id=episode_id)
        self.dt = dt
        self.num_goals = num_goals
        self.collision_threshold = collision_threshold
        self.rows = []
        self.previous_collision = False
        self.previous_action = None
        self.collision_events = 0
        self.collision_steps = 0
        self.action_deltas = []

    def add(self, state, action, depth_fresh=False, moe=None, attention=None):
        action = np.asarray(action, dtype=float)
        pos = np.asarray(state["position"], dtype=float)
        quat = np.asarray(state["quaternion"], dtype=float)
        lin = np.asarray(state["linear_velocity"], dtype=float)
        ang = np.asarray(state["angular_velocity"], dtype=float)
        joint_vel = np.asarray(state["joint_velocity"], dtype=float)
        torque = np.asarray(state["torque"], dtype=float)
        goal = np.asarray(state["goal"], dtype=float)
        goal_vector = goal[:2] - pos[:2]
        goal_direction = goal_vector / (np.linalg.norm(goal_vector) + 1e-5)
        projected_speed = np.dot(np.asarray(state["world_linear_velocity"])[:2], goal_direction)
        heading_error = np.arctan2(np.sin(state["target_yaw"] - state["yaw"]),
                                   np.cos(state["target_yaw"] - state["yaw"]))
        collision_force = float(state["base_contact_force"])
        collision = collision_force > self.collision_threshold
        self.collision_events += int(collision and not self.previous_collision)
        self.collision_steps += int(collision)
        self.previous_collision = collision
        if self.previous_action is not None:
            self.action_deltas.append(float(np.mean((action - self.previous_action) ** 2)))
        self.previous_action = action.copy()
        row = dict(self.meta, timestamp_s=(len(self.rows) + 1) * self.dt,
                   step=len(self.rows) + 1, position=pos, quaternion=quat,
                   linear_velocity=lin, angular_velocity=ang,
                   joint_position=np.asarray(state["joint_position"]),
                   joint_velocity=joint_vel, torque=torque, action=action,
                   goal=goal, command=np.asarray(state["command"]),
                   goal_index=int(state["goal_index"]),
                   roll=float(state["roll"]), pitch=float(state["pitch"]),
                   heading_error=abs(float(heading_error)),
                   goal_speed_error=abs(float(state["command"][0]) - projected_speed),
                   power_proxy=float(np.sum(np.abs(torque * joint_vel))),
                   collision=collision, base_contact_force=collision_force,
                   depth_fresh=bool(depth_fresh), moe=moe, attention=attention)
        self.rows.append(row)

    def finish(self, reason):
        if not self.rows:
            raise ValueError("episode has no recorded control steps")
        rows = self.rows
        mean = lambda key: float(np.mean([row[key] for row in rows]))
        return dict(self.meta, reason=reason, success=int(reason == "success"),
                    fallen=int(reason == "fall"), elapsed_s=len(rows) * self.dt,
                    steps=len(rows), progress=min(1., rows[-1]["goal_index"] / self.num_goals),
                    collision_events=self.collision_events,
                    collision_duration_s=self.collision_steps * self.dt,
                    max_collision_force=max(row["base_contact_force"] for row in rows),
                    mean_abs_roll=float(np.mean([abs(row["roll"]) for row in rows])),
                    mean_abs_pitch=float(np.mean([abs(row["pitch"]) for row in rows])),
                    xy_ang_vel_rms=float(np.sqrt(np.mean([np.mean(row["angular_velocity"][:2] ** 2) for row in rows]))),
                    mean_heading_error=mean("heading_error"),
                    mean_goal_speed_error=mean("goal_speed_error"),
                    action_smoothness=float(np.mean(self.action_deltas)) if self.action_deltas else 0.,
                    mean_power_proxy=mean("power_proxy"),
                    energy_proxy_j=float(sum(row["power_proxy"] for row in rows) * self.dt))


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, default=lambda item: item.tolist() if isinstance(item, np.ndarray) else float(item))
                             if isinstance(value, (list, tuple, np.ndarray, dict)) else value
                             for key, value in row.items()})


def summarize(episodes):
    groups = defaultdict(list)
    for episode in episodes:
        groups[(episode["model"], episode["terrain"], episode["level"])].append(episode)
    result = []
    for (model, terrain, level), group in sorted(groups.items()):
        successful = [row for row in group if row["success"]]
        avg = lambda key: float(np.mean([row[key] for row in group]))
        result.append(dict(model=model, terrain=terrain, level=level, episodes=len(group),
                           success_rate=avg("success"), fall_rate=avg("fallen"),
                           successful_completion_s=float(np.mean([row["elapsed_s"] for row in successful])) if successful else np.nan,
                           mean_collision_events=avg("collision_events"),
                           mean_collision_duration_s=avg("collision_duration_s"),
                           mean_max_collision_force=avg("max_collision_force"),
                           mean_abs_roll=avg("mean_abs_roll"), mean_abs_pitch=avg("mean_abs_pitch"),
                           mean_xy_ang_vel_rms=avg("xy_ang_vel_rms"),
                           mean_heading_error=avg("mean_heading_error"),
                           mean_goal_speed_error=avg("mean_goal_speed_error"),
                           mean_action_smoothness=avg("action_smoothness"),
                           mean_power_proxy=avg("mean_power_proxy")))
    return result


def plot_results(output_dir, summaries, steps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    if summaries:
        metrics = ["success_rate", "successful_completion_s", "fall_rate", "mean_collision_events",
                   "mean_collision_duration_s", "mean_max_collision_force", "mean_abs_roll", "mean_abs_pitch",
                   "mean_xy_ang_vel_rms", "mean_heading_error", "mean_goal_speed_error",
                   "mean_action_smoothness", "mean_power_proxy"]
        labels = [f'{row["terrain"]} L{row["level"]} {row["model"]}' for row in summaries]
        for metric in metrics:
            if len(labels) <= 24:
                fig, ax = plt.subplots(figsize=(max(8, len(labels) * .7), 4.5))
                ax.bar(labels, [row[metric] for row in summaries])
                ax.tick_params(axis="x", labelrotation=45)
                ax.set_ylabel(metric.replace("_", " "))
            else:
                cases = sorted({(row["terrain"], row["level"]) for row in summaries})
                models = sorted({row["model"] for row in summaries})
                values = {(row["terrain"], row["level"], row["model"]): row[metric] for row in summaries}
                matrix = np.array([[values.get((terrain, level, model), np.nan) for model in models]
                                   for terrain, level in cases])
                fig, ax = plt.subplots(figsize=(7, max(5, len(cases) * .3)))
                image = ax.imshow(matrix, aspect="auto", cmap="viridis")
                ax.set_xticks(range(len(models)), models)
                ax.set_yticks(range(len(cases)), [f"{terrain} L{level}" for terrain, level in cases], fontsize=7)
                fig.colorbar(image, ax=ax, label=metric.replace("_", " "))
            fig.tight_layout()
            fig.savefig(output_dir / f"{metric}.png", dpi=300)
            plt.close(fig)
    moe_rows = [row for row in steps if row.get("moe") is not None]
    if moe_rows:
        by_terrain = defaultdict(list)
        for row in moe_rows:
            by_terrain[(row["terrain"], row["level"])].append(row["moe"]["gate_weights"])
        keys = sorted(by_terrain)
        weights = np.array([np.mean(by_terrain[key], axis=0) for key in keys])
        fig, ax = plt.subplots(figsize=(7, max(3, len(keys) * .45)))
        image = ax.imshow(weights, aspect="auto", vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks(range(4), [f"Expert {i}" for i in range(4)])
        ax.set_yticks(range(len(keys)), [f"{a} L{b}" for a, b in keys])
        fig.colorbar(image, ax=ax, label="Mean gate weight")
        fig.tight_layout()
        fig.savefig(output_dir / "moe_terrain_heatmap.png", dpi=300)
        plt.close(fig)
        first = moe_rows[0]
        curve = [row for row in moe_rows if row["terrain"] == first["terrain"] and row["level"] == first["level"]
                 and row["env_id"] == first["env_id"] and row["episode_id"] == first["episode_id"]]
        fig, ax = plt.subplots(figsize=(8, 4))
        for expert in range(4):
            ax.plot([row["timestamp_s"] for row in curve],
                    [row["moe"]["gate_weights"][expert] for row in curve], label=f"Expert {expert}")
        ax.set(xlabel="Time (s)", ylabel="Gate weight")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "moe_weight_time.png", dpi=300)
        plt.close(fig)
    attention_rows = [row for row in steps if row.get("attention") is not None]
    if attention_rows:
        weights = np.mean([row["attention"] for row in attention_rows], axis=0)
        fig, ax = plt.subplots(figsize=(8, 3.5))
        image = ax.imshow(weights[:, 0, :], aspect="auto", cmap="viridis")
        ax.set(xlabel="Scandot row token (11 samples each)", ylabel="Attention head")
        ax.set_yticks(range(weights.shape[0]))
        fig.colorbar(image, ax=ax)
        fig.tight_layout()
        fig.savefig(output_dir / "teacher_attention.png", dpi=300)
        plt.close(fig)
