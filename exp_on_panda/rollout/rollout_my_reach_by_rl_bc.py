from pathlib import Path
import sys
import argparse
import re
import gymnasium as gym
import panda_gym
import numpy as np
import pandas as pd
from gymnasium.envs.registration import register
from gymnasium.wrappers import FlattenObservation

import os

PROJECT_ROOT_DIR = Path().absolute().parent
PROJECT_ROOT_DIR

if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from rollout_by_policy import rollout_by_goal_with_policy
from utils.sb3_env_wrappers import ScaledObservationWrapper
from utils.load_data import load_data
from models.sb3_model import PPOWithBCLoss
from configs.load_config import load_config


def main(base_data_file, current_iter):
    register(
        id=ENV_NAME,
        entry_point=f"my_reach_env:MyPandaReachEnv",
        kwargs={"reward_type": "sparse", "control_type": "ee", "goal_range": goal_range, "distance_threshold": distance_threshold},
        max_episode_steps=50,
    )

    data_file: Path = PROJECT_ROOT_DIR / 'exp_on_panda' / BC_EXPERT_DATA_DIR
    _, _, _, _, _, obs_scaler = load_data(data_file)

    env = gym.make(ENV_NAME)
    env = ScaledObservationWrapper(env=FlattenObservation(env), scaler=obs_scaler)

    # load policy
    policy_save_dir = PROJECT_ROOT_DIR / "exp_on_panda" / "checkpoints" / "rl" / RL_EXPERIMENT_NAME
    algo_ppo = PPOWithBCLoss.load(str((policy_save_dir / "best_model").absolute()))

    # Load base data (iter 1: PID data, iter n>1: iter n-1's rollout data)
    if not base_data_file.exists():
        raise FileNotFoundError(f"Base data file not found: {base_data_file}. "
                              f"For iter {current_iter}, expected file: {base_data_file.name}")
    
    base_df = pd.read_csv(base_data_file)
    data_source = "PID" if current_iter == 1 else f"iter {current_iter - 1}"
    print(f"Loaded base data from {data_source}: {len(base_df):,} rows (file: {base_data_file.name})")
    
    # Group base data by goal to process each goal once
    base_goals = base_df[['s_g_x', 's_g_y', 's_g_z']].drop_duplicates()
    print(f"Processing {len(base_goals)} unique goals")
    
    target_total_rows = len(base_df)  # Target: match original data size
    print(f"Target total rows: {target_total_rows:,}")
    
    # Initialize result list to store rows
    result_rows = []
    
    total_goals = len(base_goals)
    MAX_TRIALS_PER_GOAL = 3  # Number of trajectory generations per goal per iteration
    
    # Repeat the entire process until we reach target data size
    iteration = 0
    while len(result_rows) < target_total_rows:
        iteration += 1
        current_size = len(result_rows)
        print(f"\n{'='*60}")
        print(f"Iteration {iteration}: Processing all goals...")
        print(f"Progress: [{current_size:,}/{target_total_rows:,}] rows")
        print(f"{'='*60}")
        
        iteration_replaced_cnt = 0
        iteration_kept_cnt = 0
        
        # Process each goal
        for goal_idx, (_, goal_row) in enumerate(base_goals.iterrows()):
            # Stop if we've reached target size
            if len(result_rows) >= target_total_rows:
                break
                
            goal = np.array([goal_row['s_g_x'], goal_row['s_g_y'], goal_row['s_g_z']])
            goal_key = (float(goal[0]), float(goal[1]), float(goal[2]))
            
            # Find all rows with this goal in original data (the entire trajectory)
            goal_mask = (base_df['s_g_x'] == goal_key[0]) & \
                       (base_df['s_g_y'] == goal_key[1]) & \
                       (base_df['s_g_z'] == goal_key[2])
            original_traj_rows = base_df[goal_mask]
            original_steps = len(original_traj_rows)
            
            # Determine if original trajectory is successful (not too long, less than 45 steps)
            original_is_success = original_steps < 45
            
            # Generate new trajectories for this goal
            best_single_traj = None
            best_single_terminated = False
            best_single_steps = float('inf')
            
            # Generate multiple trajectories and find the best one
            for trial in range(MAX_TRIALS_PER_GOAL):
                current_size = len(result_rows)
                progress_info = f" | Progress: [{current_size:,}/{target_total_rows:,}]"
                terminated, truncated, new_traj = rollout_by_goal_with_policy(env=env, goal=goal, policy=algo_ppo.policy, progress_info=progress_info)
                new_traj_steps = len(new_traj['s_x'])
                
                # Track the best single trajectory
                if terminated:
                    if best_single_traj is None or not best_single_terminated or new_traj_steps < best_single_steps:
                        best_single_traj = new_traj
                        best_single_terminated = True
                        best_single_steps = new_traj_steps
                elif not best_single_terminated and (best_single_traj is None or new_traj_steps < best_single_steps):
                    best_single_traj = new_traj
                    best_single_terminated = False
                    best_single_steps = new_traj_steps
            
            # Decide whether to replace or keep original
            should_replace = False
            if best_single_traj is not None:
                if best_single_terminated and not original_is_success:
                    # 새 trajectory가 성공하고 원본이 실패 -> 교체
                    should_replace = True
                elif best_single_terminated and original_is_success:
                    # 둘 다 성공하면 새 trajectory가 더 짧으면 교체
                    if best_single_steps < original_steps:
                        should_replace = True
            
            # Add trajectory to result (either new or original)
            if should_replace:
                # Use best single trajectory
                best_single_df = pd.DataFrame(best_single_traj)
                result_rows.extend(best_single_df.to_dict('records'))
                iteration_replaced_cnt += 1
            else:
                # Keep original trajectory
                result_rows.extend(original_traj_rows.to_dict('records'))
                iteration_kept_cnt += 1
            
            # Progress update
            if (goal_idx + 1) % 100 == 0:
                current_total = len(result_rows)
                print(f"  Processed {goal_idx+1:,}/{total_goals:,} goals | Progress: [{current_total:,}/{target_total_rows:,}] rows (replaced: {iteration_replaced_cnt}, kept: {iteration_kept_cnt})")
        
        current_size = len(result_rows)
        print(f"\nIteration {iteration} complete:")
        print(f"  Replaced: {iteration_replaced_cnt} goals")
        print(f"  Kept: {iteration_kept_cnt} goals")
        print(f"  Progress: [{current_size:,}/{target_total_rows:,}] rows")
        
        # If we haven't reached target, continue to next iteration
        if current_size < target_total_rows:
            remaining = target_total_rows - current_size
            print(f"  Need {remaining:,} more rows, continuing to next iteration...")
    
    # Trim to exact target size if we exceeded it
    if len(result_rows) > target_total_rows:
        result_rows = result_rows[:target_total_rows]
        print(f"\nTrimmed to target size: {len(result_rows):,} rows")
    
    # Convert to DataFrame
    result_df = pd.DataFrame(result_rows)
    
    print(f"\n{'='*60}")
    print(f"Final Summary:")
    print(f"  Total iterations: {iteration}")
    print(f"  Total goals processed per iteration: {total_goals:,}")
    print(f"  Final data: {len(result_df):,} rows (original: {len(base_df):,} rows)")
    print(f"{'='*60}")

    os.makedirs(os.path.dirname(csv_save_name), exist_ok=True)
    result_df.to_csv(csv_save_name, index=False)
    print(f"Saved to {csv_save_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pass in config file")
    parser.add_argument("--config_file_name", type=str, help="Config file name", default="configs/iter_1/seed1/reacher_annealing.json")
    args = parser.parse_args()

    custom_config = load_config(args.config_file_name)

    ENV_NAME = custom_config["env"]["name"]
    RL_EXPERIMENT_NAME = custom_config["rl_bc"]["experiment_name"]
    BC_EXPERT_DATA_DIR = custom_config["bc"]["data_cache_dir"]

    # Extract iteration and seed numbers from config file path (e.g., "configs/iter_1/seed1/..." -> iter=1, seed=1)
    iter_match = re.search(r'iter_(\d+)', args.config_file_name)
    seed_match = re.search(r'seed(\d+)', args.config_file_name)
    current_iter = int(iter_match.group(1)) if iter_match else 1
    seed_suffix = f'_{seed_match.group(1)}' if seed_match else ''
    seed_num = seed_match.group(1) if seed_match else '1'

    # Local rollout grid parameters (kept as constants; add to config if needed)
    goal_range = 0.3
    distance_threshold = 0.01
    csv_save_name = 'rollout/cache/' + 'myreach_from_' + RL_EXPERIMENT_NAME.split('/')[0] + '_rl_bc' + seed_suffix + '.csv'
    
    # Determine which data file to use as base:
    # - iter 1: use PID data (myreach_pid_speed_1.5.csv)
    # - iter n (n > 1): use iter n-1's rollout data (myreach_from_iter_{n-1}_rl_bc_{seed}.csv)
    if current_iter == 1:
        base_data_file = PROJECT_ROOT_DIR / 'exp_on_panda' / 'rollout' / 'cache' / 'myreach_pid_speed_1.5.csv'
    else:
        prev_iter = current_iter - 1
        base_data_file = PROJECT_ROOT_DIR / 'exp_on_panda' / 'rollout' / 'cache' / f'myreach_from_iter_{prev_iter}_rl_bc_{seed_num}.csv'

    main(base_data_file, current_iter)
