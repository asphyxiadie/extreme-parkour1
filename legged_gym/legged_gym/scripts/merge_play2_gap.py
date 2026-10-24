"""Use the 0.5 m single-robot MoE gap run in the four-terrain comparison.

This module reads existing CSV files only. It never changes checkpoint, video, or
per-run CSV files. The merged comparison keeps the source of every row.
"""

import argparse
import csv
import json
import math
import os
from pathlib import Path


def read_csv(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    if not rows:
        raise ValueError(f'No rows to write: {path}')
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def mean(rows, key):
    return sum(float(row[key]) for row in rows) / len(rows)


def reached_gap_finish(step, metadata):
    x_margin = float(metadata.get('gap_finish_x_margin_m', 0.5))
    y_tolerance = float(metadata.get('gap_finish_y_tolerance_m', 0.6))
    min_height = float(metadata.get('gap_finish_min_base_height_m', 0.2))
    return (float(step['position_x']) >= float(step['final_goal_x']) - x_margin
            and abs(float(step['position_y']) - float(step['final_goal_y'])) <= y_tolerance
            and float(step['position_z']) >= min_height
            and abs(float(step['roll_rad'])) <= 1.5
            and abs(float(step['pitch_rad'])) <= 1.5)


def finish_at_first_arrival(episode, steps, metadata):
    """Cut metrics at the first stable arrival, before later waiting/falling."""
    arrival = next((i for i, step in enumerate(steps)
                    if reached_gap_finish(step, metadata)), None)
    result = dict(episode)
    result['original_reason'] = episode['reason']
    result['original_elapsed_s'] = episode['elapsed_s']
    result['reclassified_by_merge'] = 0
    result['video_contains_post_success'] = 0
    if arrival is None:
        return result
    prefix = steps[:arrival + 1]
    final = prefix[-1]
    result.update(reason='success', success=1, success_via='gap_finish',
                  elapsed_s=final['time_s'], steps=len(prefix), progress=1.0,
                  final_goal_distance_m=final['final_goal_distance_m'],
                  mean_command_vx=mean(prefix, 'command_vx'),
                  mean_actual_vx=mean(prefix, 'actual_vx'),
                  mean_abs_speed_error=sum(abs(float(r['command_vx']) - float(r['actual_vx']))
                                           for r in prefix) / len(prefix),
                  collision_events=sum(int(r['collision_start']) for r in prefix),
                  collision_rate=mean(prefix, 'collision'),
                  collision_duration_s=sum(int(r['collision']) for r in prefix)
                                       * float(metadata['control_dt_s']),
                  max_body_contact_force_n=max(float(r['body_contact_force_n']) for r in prefix),
                  mean_abs_roll_rad=sum(abs(float(r['roll_rad'])) for r in prefix) / len(prefix),
                  mean_abs_pitch_rad=sum(abs(float(r['pitch_rad'])) for r in prefix) / len(prefix),
                  rms_ang_vel_xy=math.sqrt(sum(float(r['ang_vel_x']) ** 2
                                                + float(r['ang_vel_y']) ** 2
                                                for r in prefix) / len(prefix)))
    changes = [float(r['action_delta_abs_mean']) for r in prefix
               if r['action_delta_abs_mean']]
    result['action_smoothness'] = sum(changes) / len(changes) if changes else float('nan')
    result['reclassified_by_merge'] = int(episode['reason'] != 'success')
    result['video_contains_post_success'] = int(arrival < len(steps) - 1)
    return result


def gap_result(run_dir):
    metadata = json.loads((run_dir / 'metadata.json').read_text())
    if (metadata.get('model') != 'moe' or not metadata.get('gap_only')
            or not math.isclose(float(metadata.get('parkour_gap_depth_m', -1)), 0.5)):
        raise ValueError(f'Expected a MoE gap-only run with 0.5 m depth: {run_dir}')
    episodes = read_csv(run_dir / 'episodes.csv')
    summary = read_csv(run_dir / 'summary.csv')
    if len(summary) != 1 or summary[0]['terrain'] != 'parkour_gap':
        raise ValueError(f'Expected one parkour_gap summary row: {run_dir}')
    if len(episodes) != int(summary[0]['episodes']):
        raise ValueError('Episode count does not match the new gap summary')
    by_episode = {}
    for step in read_csv(run_dir / 'steps.csv'):
        key = (step['env_id'], step['episode_id'])
        by_episode.setdefault(key, []).append(step)
    corrected = []
    for episode in episodes:
        key = (episode['env_id'], episode['episode_id'])
        steps = by_episode.get(key)
        if not steps:
            raise ValueError(f'Missing steps for gap episode {key}')
        corrected.append(finish_at_first_arrival(episode, steps, metadata))
    row = dict(summary[0])
    successes = [episode for episode in corrected if int(episode['success'])]
    row.update(episodes=len(corrected), successes=len(successes),
               success_rate=len(successes) / len(corrected),
               fall_rate=sum(episode['reason'] == 'fall' for episode in corrected) / len(corrected),
               mean_success_time_s=(mean(successes, 'elapsed_s') if successes else float('nan')),
               mean_collision_rate=mean(corrected, 'collision_rate'),
               mean_collision_events=mean(corrected, 'collision_events'),
               mean_actual_vx=mean(corrected, 'mean_actual_vx'),
               mean_abs_speed_error=mean(corrected, 'mean_abs_speed_error'),
               mean_abs_roll_rad=mean(corrected, 'mean_abs_roll_rad'),
               mean_abs_pitch_rad=mean(corrected, 'mean_abs_pitch_rad'),
               mean_rms_ang_vel_xy=mean(corrected, 'rms_ang_vel_xy'),
               mean_action_smoothness=mean(corrected, 'action_smoothness'),
               reclassified_episodes=sum(int(e['reclassified_by_merge']) for e in corrected))
    return row, corrected, metadata


def annotate(row, run_dir, metadata, comparison_root):
    result = dict(row)
    result['source_env_id'] = result['env_id']
    if result['model'] == 'moe' and result['terrain'] == 'parkour_gap':
        result['env_id'] = 3  # The gap slot in the original four-robot comparison.
    result['source_run'] = os.path.relpath(run_dir, comparison_root)
    result['checkpoint'] = metadata.get('checkpoint', '')
    result['seed'] = metadata.get('seed', '')
    result['difficulty'] = metadata.get('fixed_difficulty', '')
    result['gap_depth_m'] = (metadata.get('parkour_gap_depth_m', '')
                             if result['terrain'] == 'parkour_gap' else '')
    result['success_rule'] = metadata.get('success_rule', '')
    if 'video' in result:
        result['video_path'] = os.path.relpath(run_dir / result['video'], comparison_root)
    return result


def merge_gap_comparison(comparison_root, gap_root):
    comparison_root, gap_root = Path(comparison_root).expanduser(), Path(gap_root).expanduser()
    new_run = gap_root / 'moe'
    gap_summary, gap_episodes, gap_metadata = gap_result(new_run)
    summary_rows, episode_rows = [], []
    for model in ('teacher', 'moe'):
        run_dir = comparison_root / model
        metadata = json.loads((run_dir / 'metadata.json').read_text())
        for row in read_csv(run_dir / 'summary.csv'):
            if model == 'moe' and row['terrain'] == 'parkour_gap':
                continue
            summary_rows.append(annotate(row, run_dir, metadata, comparison_root))
        for row in read_csv(run_dir / 'episodes.csv'):
            if model == 'moe' and row['terrain'] == 'parkour_gap':
                continue
            episode_rows.append(annotate(row, run_dir, metadata, comparison_root))
    summary_rows.append(annotate(gap_summary, new_run, gap_metadata, comparison_root))
    episode_rows.extend(annotate(row, new_run, gap_metadata, comparison_root)
                        for row in gap_episodes)
    if len({(r['model'], r['terrain']) for r in summary_rows}) != len(summary_rows):
        raise ValueError('Duplicate model/terrain rows in comparison')
    write_csv(comparison_root / 'comparison.csv', summary_rows)
    write_csv(comparison_root / 'comparison_episodes.csv', episode_rows)
    print(f"Updated {comparison_root / 'comparison.csv'} with MoE gap 0.5 m: "
          f"{gap_summary['successes']}/{gap_summary['episodes']} successes, "
          f"{gap_summary['reclassified_episodes']} earlier arrivals reclassified")
    return summary_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', default='~/gpufree-data/parkour-data',
                        help='Directory containing teacher/ and moe/ four-terrain runs')
    parser.add_argument('--gap-run', default='~/gpufree-data/parkour-data/moe-gap-0p5',
                        help='Directory containing moe/ gap-only run')
    args = parser.parse_args()
    merge_gap_comparison(args.base, args.gap_run)


if __name__ == '__main__':
    main()
