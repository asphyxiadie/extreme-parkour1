"""CPU-only checks for evaluation outcomes, aggregation, and artifacts."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


MODULE = Path(__file__).resolve().parents[1] / "legged_gym/legged_gym/utils/evaluation_metrics.py"
spec = importlib.util.spec_from_file_location("evaluation_metrics", MODULE)
metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics)


def state(force, goal_index=0):
    return dict(position=np.array([0., 0., .4]), quaternion=np.array([0., 0., 0., 1.]),
                linear_velocity=np.array([.5, 0., 0.]), world_linear_velocity=np.array([.5, 0., 0.]),
                angular_velocity=np.array([.2, .4, 0.]), joint_position=np.zeros(12),
                joint_velocity=np.ones(12), torque=np.ones(12) * 2,
                goal=np.array([1., 0., 0.]), command=np.array([.6, 0., 0., 0.]),
                goal_index=goal_index, roll=.1, pitch=-.2, yaw=0., target_yaw=0.,
                base_contact_force=force)


class EvaluationMetricsTest(unittest.TestCase):
    def test_goal_timeout_and_fall_are_distinct(self):
        self.assertEqual(metrics.termination_reason(8, 8, False, True, True), "success")
        self.assertEqual(metrics.termination_reason(3, 8, False, True, True), "timeout")
        self.assertEqual(metrics.termination_reason(8, 8, True, True, True), "fall")

    def test_episode_metrics_and_artifacts(self):
        episode = metrics.EpisodeAccumulator("moe", "parkour_gap", 4, 19, 0, 0, .02, 8, 1.)
        episode.add(state(0.), np.zeros(12))
        episode.add(state(2.), np.ones(12))
        episode.add(state(3., 8), np.ones(12))
        result = episode.finish("success")
        self.assertEqual(result["collision_events"], 1)
        self.assertAlmostEqual(result["collision_duration_s"], .04)
        self.assertAlmostEqual(result["max_collision_force"], 3.)
        self.assertAlmostEqual(result["action_smoothness"], .5)
        self.assertAlmostEqual(result["mean_power_proxy"], 24.)
        self.assertAlmostEqual(result["energy_proxy_j"], 1.44)
        summary = metrics.summarize([result])
        self.assertEqual(summary[0]["success_rate"], 1.)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            episode.rows[0]["moe"] = dict(gate_weights=np.full(4, .25), expert_l2=np.ones(4),
                                           weighted_l2=np.full(4, .25), residual_scale=.1,
                                           router_entropy=np.log(4), top1=0)
            episode.rows[0]["attention"] = np.full((4, 1, 12), 1 / 12)
            metrics.write_csv(path / "episodes.csv", [result])
            metrics.write_csv(path / "steps.csv", episode.rows)
            metrics.plot_results(path, summary, episode.rows)
            self.assertTrue((path / "episodes.csv").is_file())
            self.assertTrue((path / "success_rate.png").is_file())
            self.assertTrue((path / "moe_weight_time.png").is_file())
            self.assertTrue((path / "moe_terrain_heatmap.png").is_file())
            self.assertTrue((path / "teacher_attention.png").is_file())


if __name__ == "__main__":
    unittest.main()
