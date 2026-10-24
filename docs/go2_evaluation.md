# Go2 Play Evaluation

Run from the repository root in the Isaac Gym environment:

```bash
python legged_gym/legged_gym/scripts/play_evaluation.py \
  --model all --terrain parkour_gap --level 4 --seed 1234 \
  --episodes 20 --num-envs 4 --max-steps 3000 \
  --teacher-run legged_gym/logs/go2-attn-01 --teacher-checkpoint 14500 \
  --student-run legged_gym/logs/go2-vision-01 --student-checkpoint 9500 \
  --moe-run legged_gym/logs/go2-moe-01 --moe-checkpoint 14000 \
  --record-steps --record-moe --record-attention \
  --output legged_gym/evaluation/go2-gap-l4
```

Use `--model teacher|student|moe` and `--checkpoint N` for one policy. `--terrain all`
tests all `parkour*` terrain names; `--level all` tests levels 0–9. Each model,
terrain and level receives `--episodes` **completed** episodes. Checkpoints are
loaded with `strict=True`; the original student uses an evaluation-only legacy
CNN+GRU class, while the MoE student uses the MoE class. The local teacher directory
currently contains `model_14500.pt` as its highest numbered file, not `model_15000.pt`.
Specify `--teacher-checkpoint 15000` when that file is available.

The evaluation builds the same terrain from the same seed for each policy and fixes
the terrain level and speed command (0.6 m/s). It disables curriculum updates,
observation noise, pushes and physical randomization. Joint and base resets are
deterministic. The only policy-specific environment difference is that students
render a depth camera. GPU physics may still be nondeterministic, so trajectories
will diverge after the first action. The run uses a fixed one-control-step action
delay for all policies. The result metadata records the seed, selected checkpoint,
terrain, level and camera update interval.

The terrain generator uses row `i` to set difficulty `i / 9`. For the default
`parkour*` suite, the requested terrain name matches the generated terrain class.
For other terrain keys, consult `terrain_class_id` in the CSV; the upstream terrain
generator combines some named slope and stair categories internally.

## Files

- `episodes.csv`: one row per completed episode, keyed by model, terrain, level,
  terrain class ID, env ID and episode ID. Includes outcome, time, progress and
  all aggregate metrics.
- `summary.csv`: group means by model, terrain and level; completion time is
  averaged **only over successes** and is NaN if there are none.
- `steps.csv`: one row per control step with `--record-steps`. Without that option,
  it contains only diagnostic steps when MoE or attention recording is enabled.
  Arrays and nested diagnostics are JSON encoded in their CSV cells.
- `records.npz`: numeric trajectory arrays, identifiers and diagnostic arrays.
  `moe_step_index` and `attention_step_index` map sparse diagnostics to step rows.
  Load with `np.load(path, allow_pickle=False)`.
- `metadata.json`: reproducibility parameters and checkpoint paths.
- 300 DPI PNGs: success, completion time, fall, collision, attitude, tracking,
  smoothness and power comparisons; `moe_weight_time.png`,
  `moe_terrain_heatmap.png`, and `teacher_attention.png` when enabled.

## Metric definitions

Success means `cur_goal_idx >= num_goals` without a simultaneous fall. A fall means
`|roll| > 1.5 rad`, `|pitch| > 1.5 rad`, or root height `< -0.25 m`. The environment
also marks success as a timeout; the evaluator uses the goal index saved **before
reset** to separate success from actual time limit. Other reset reasons are retained
as `other`. Progress is `min(1, final_goal_index / num_goals)`. Elapsed time is
control steps times `dt`.

Base collision uses the norm of the configured `termination_contact_indices` force
and a configurable threshold (`--collision-threshold`, 1 N by default). Events count
threshold crossings; duration counts colliding control steps times `dt`. Maximum
collision force is the largest measured base force. Roll and pitch report separate
means of absolute radians. XY angular velocity RMS is
`sqrt(mean((omega_x² + omega_y²)/2))`. Heading error is the mean absolute wrapped
angle between target and base yaw. Goal-speed error is the mean absolute difference
between commanded forward speed and world XY speed projected toward the current
goal. Action smoothness is mean squared per-joint difference of consecutive policy
actions. Power proxy is `sum(abs(torque_j * joint_velocity_j))`, averaged per step;
the energy proxy integrates it over time. Torque is the last PD torque computed
inside each control step, not a measured motor power trace.

The 132 Scandots are row-major `12×11`; each row is one attention token. The
attention array is `[step, head=4, query=1, token=12]`. Individual samples within
a token cannot be resolved from these weights. Attention is recomputed for logging
using the trained Q/K projection weights; the actor action still uses the original
forward path. MoE diagnostics store four gate weights, raw expert L2 norms,
weighted expert L2 norms, learned residual scale, entropy and top-1 index. The
depth encoder is called on the initial frame and then only when the camera delivers
a frame (normally every five control steps); sparse MoE rows correspond to those
actual encoder forwards. GRU hidden state and cached latent are cleared per reset.
