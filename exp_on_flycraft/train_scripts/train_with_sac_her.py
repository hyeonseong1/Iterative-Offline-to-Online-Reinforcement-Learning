import gymnasium as gym
import numpy as np
from pathlib import Path
import logging
from time import time
import argparse
import os
import sys
import torch as th
import warnings

# Suppress gymnasium warnings
warnings.filterwarnings("ignore", message=".*env.compute_reward to get variables from other wrappers is deprecated.*")


from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.common.vec_env import SubprocVecEnv, VecCheckNan
from stable_baselines3.common.logger import configure, Logger
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

import flycraft
from flycraft.env import FlyCraftEnv
from flycraft.utils_common.load_config import load_config

PROJECT_ROOT_DIR = Path(__file__).parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from utils_my.sb3.my_wrappers import ScaledObservationWrapper, ScaledActionWrapper
from utils_my.sb3.vec_env_helper import get_vec_env
from utils_my.sb3.my_eval_callback import MyEvalCallback
from utils_my.sb3.my_schedule import linear_schedule

def train():
    
    # Setup Logger
    log_dir = PROJECT_ROOT_DIR / "logs" / "rl" / EXPERIMENT_NAME
    sb3_logger: Logger = configure(folder=str(log_dir.absolute()), format_strings=['stdout', 'log', 'csv', 'tensorboard'])

    # Environment Config
    env_config_dict_in_training = {
        "num_process": ROLLOUT_PROCESS_NUM,
        "seed": SEED,
        "config_file": str(PROJECT_ROOT_DIR / "configs" / "env" / train_config["env"].get("config_file", "env_config_for_ppo.json")),
        "custom_config": {"debug_mode": False, "flag_str": "Train"}
    }
    
    # Initialize Vector Environment
    vec_env = VecCheckNan(get_vec_env(
        **env_config_dict_in_training
    ))

    # Initialize Evaluation Environment
    env_config_dict_in_eval = env_config_dict_in_training.copy()
    env_config_dict_in_eval.update({
        "num_process": EVALUATE_PROCESS_NUM,
        "custom_config": {"debug_mode": False, "flag_str": "Evaluate"}
    })
    eval_env = VecCheckNan(get_vec_env(
        **env_config_dict_in_eval
    ))

    # Initialize SAC Algorithm with HER
    model = SAC(
        POLICY,
        vec_env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=4,
            goal_selection_strategy="future",
        ),
        verbose=1,
        learning_rate=linear_schedule(LEARNING_RATE),
        buffer_size=BUFFER_SIZE,
        learning_starts=LEARNING_STARTS,
        batch_size=BATCH_SIZE,
        tau=TAU,
        gamma=GAMMA,
        train_freq=TRAIN_FREQ,
        gradient_steps=GRADIENT_STEPS,
        action_noise=None, # SAC uses stochastic policy
        ent_coef=ENT_COEF,
        seed=SEED,
    )
    
    model.set_logger(sb3_logger)

    # Checkpoint Callback
    checkpoint_dir = PROJECT_ROOT_DIR / "checkpoints" / "sac_her" / EXPERIMENT_NAME
    # checkpoint_callback = CheckpointCallback(save_freq=max(10000 // ROLLOUT_PROCESS_NUM, 1), save_path=str(checkpoint_dir), name_prefix="sac_her_model")

    # Evaluation Callback
    eval_callback = MyEvalCallback(
        eval_env,
        best_model_save_path=str(checkpoint_dir),
        log_path=str(log_dir),
        eval_freq=EVALUATE_FREQUENCE,
        n_eval_episodes=EVALUATE_NUMS_IN_EVALUATION * EVALUATE_PROCESS_NUM,
        deterministic=True,
        render=False
    )

    # Train
    print(f"Start training SAC+HER for {TOTAL_TIMESTEPS} steps...")
    start_time = time()
    # model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=[checkpoint_callback, eval_callback])
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=[eval_callback])
    print(f"Training finished in {time() - start_time:.2f} seconds.")

    # Save final model as 'best_model.zip' if not saved by callback or just as final
    final_save_path = checkpoint_dir / "final_model"
    model.save(final_save_path)
    print(f"Model saved to {final_save_path}")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="SAC HER Training")
    parser.add_argument("--config-file-name", type=str, help="Configuration file name", default="sac_her_default.json")
    args = parser.parse_args()

    config_path = Path(os.getcwd()) / "configs" / "train" / args.config_file_name
    if not config_path.exists():
        # fallback to looking inside exp_on_flycraft if run from root
       config_path = PROJECT_ROOT_DIR / "configs" / "train" / args.config_file_name
       
    train_config = load_config(config_path)

    # Parse Config
    SAC_CONFIG = train_config["sac"]
    EXPERIMENT_NAME = SAC_CONFIG["experiment_name"]
    SEED = SAC_CONFIG.get("seed", 42)
    TOTAL_TIMESTEPS = int(float(SAC_CONFIG.get("total_timesteps", 1e6)))
    POLICY = SAC_CONFIG.get("policy", "MultiInputPolicy")
    LEARNING_RATE = SAC_CONFIG.get("learning_rate", 3e-4)
    BUFFER_SIZE = int(float(SAC_CONFIG.get("buffer_size", 1000000)))
    BATCH_SIZE = SAC_CONFIG.get("batch_size", 256)
    ENT_COEF = SAC_CONFIG.get("ent_coef", "auto")
    GAMMA = SAC_CONFIG.get("gamma", 0.99)
    TAU = SAC_CONFIG.get("tau", 0.005)
    TRAIN_FREQ = SAC_CONFIG.get("train_freq", 1)
    GRADIENT_STEPS = SAC_CONFIG.get("gradient_steps", 1)
    LEARNING_STARTS = int(float(SAC_CONFIG.get("learning_starts", 100)))
    
    ROLLOUT_PROCESS_NUM = SAC_CONFIG.get("rollout_process_num", 1)
    EVALUATE_PROCESS_NUM = SAC_CONFIG.get("evaluate_process_num", 1)
    EVALUATE_NUMS_IN_EVALUATION = SAC_CONFIG.get("evaluate_nums_in_evaluation", 10)
    EVALUATE_FREQUENCE = SAC_CONFIG.get("evaluate_frequence", 10000)

    train()
