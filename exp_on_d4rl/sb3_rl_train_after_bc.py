import gym
import d4rl
import numpy as np
from pathlib import Path
import logging
from time import time
from copy import deepcopy
import argparse
import sys
import torch as th

from stable_baselines3.common.vec_env import SubprocVecEnv, VecCheckNan, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure, Logger
from stable_baselines3.common.callbacks import EvalCallback

PROJECT_ROOT_DIR = Path(__file__).parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from models.sb3_model import PPOWithBCLoss
from models.simba_policy import SimBaActorCriticPolicy
from configs.load_config import load_config
from utils.sb3_schedule import linear_schedule
from utils.sb3_env_utils import make_env


parser = argparse.ArgumentParser(description="Pass in config file")
parser.add_argument("--config_file_name", type=str, help="Config file name", default="iter_1/seed1/kl_1e-1/medium_halfcheetah_kl1e-1.json")
args = parser.parse_args()

custom_config = load_config(args.config_file_name)

ENV_NAME = custom_config["env"]["name"]
ENV_NORMALIZE = custom_config["env"].get("normalize", False)

BC_EXPERIMENT_NAME = custom_config["bc"]["experiment_name"]
BC_POLICY_FILE_NAME = custom_config["bc"]["policy_file_save_name"]
BC_POLICY_AFTER_VALUE_HEAD_TRAINED_FILE_NAME = custom_config["bc"]["policy_after_value_head_trained_file_save_name"]
BC_EXPERT_DATA_DIR = custom_config["bc"].get("data_cache_dir", "cache")

RL_EXPERIMENT_NAME = custom_config["rl_bc"]["experiment_name"]
SEED = custom_config["rl_bc"]["seed"]
SEED_FOR_LOAD_ALGO = custom_config["rl_bc"]["seed_for_load_algo"]
NET_ARCH = custom_config["rl_bc"]["net_arch"]
PPO_BATCH_SIZE = custom_config["rl_bc"]["batch_size"]
GAMMA = custom_config["rl_bc"]["gamma"]
GAE_LAMBDA = custom_config["rl_bc"]["gae_lambda"]
ACTIVATE_VALUE_HEAD_TRAIN_STEPS = custom_config["rl_bc"]["activate_value_head_train_steps"]
RL_TRAIN_STEPS = custom_config["rl_bc"]["train_steps"]
RL_ENT_COEF = custom_config["rl_bc"].get("ent_coef", 1e-2)
RL_LR_RATE = custom_config["rl_bc"].get("lr", 1e-4)
ROLLOUT_PROCESS_NUM = custom_config["rl_bc"]["rollout_process_num"]
N_STEPS = custom_config["rl_bc"]["n_steps"]
N_EPOCHS = custom_config["rl_bc"]["n_epochs"]
KL_WITH_BC_MODEL_COEF = custom_config["rl_bc"]["kl_with_bc_model_coef"]
KL_ANNEALING = custom_config["rl_bc"].get("kl_annealing", False)
EVAL_FREQ = custom_config["rl_bc"]["eval_freq"]
# KL decay config (start -> end), default previous 0.2 -> 0.01
KL_DECAY_START = custom_config["rl_bc"].get("kl_decay_start", 0.2)
KL_DECAY_END = custom_config["rl_bc"].get("kl_decay_end", 0.01)

np.seterr(all="raise")  # Check for nan

def kl_decay_schedule(progress_remaining: float) -> float:
    """
    KL coef decays linearly from KL_DECAY_START to KL_DECAY_END.
    progress_remaining: 1.0 -> 0.0 during training.
    """
    return KL_DECAY_END + progress_remaining * (KL_DECAY_START - KL_DECAY_END)

def get_ppo_algo(env):
    policy_kwargs = dict(
        full_std=True,  # Use state dependent exploration
        # squash_output=True,  # Use state dependent exploration
        net_arch=dict(
            pi=NET_ARCH,
            vf=deepcopy(NET_ARCH)
        ),
        activation_fn=th.nn.Tanh,
        ortho_init=True,
        optimizer_class=th.optim.Adam,
        optimizer_kwargs={
            "eps": 1e-5
        }
    )

    return PPOWithBCLoss(
        policy=SimBaActorCriticPolicy, 
        env=env, 
        seed=SEED,
        # KL coef: decay 0.2 -> 0.01 linearly over training
        kl_coef_with_bc=kl_decay_schedule,
        target_kl_with_bc=None,
        batch_size=PPO_BATCH_SIZE,  # PPO Mini Batch Size, amount of data used per PPO update
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        ent_coef=RL_ENT_COEF,
        n_steps=N_STEPS,  # Number of steps sampled per environment during sampling, total data collected per PPO training is n_steps * num_envs
        n_epochs=N_EPOCHS,  # Number of times sampled data is reused during training
        policy_kwargs=policy_kwargs,
        use_sde=True,  # Use state dependent exploration,
        normalize_advantage=True,
        learning_rate=linear_schedule(RL_LR_RATE),
    )


def train():
    
    sb3_logger: Logger = configure(folder=str((PROJECT_ROOT_DIR / "logs" / "rl" / RL_EXPERIMENT_NAME).absolute()), format_strings=['stdout', 'log', 'csv', 'tensorboard'])

    # 1. Prepare environment

    # Environment used for training
    vec_env = SubprocVecEnv([make_env(env_id=ENV_NAME, rank=i, scale_obs=True, expert_data_dir=BC_EXPERT_DATA_DIR) for i in range(ROLLOUT_PROCESS_NUM)])
    # Test environment used by evaluate_policy
    env_num_used_in_eval = 10
    eval_env = SubprocVecEnv([make_env(env_id=ENV_NAME, rank=i, scale_obs=True, expert_data_dir=BC_EXPERT_DATA_DIR) for i in range(env_num_used_in_eval)])
    # Test environment used in callback function
    env_num_used_in_callback = 10
    eval_env_in_callback = SubprocVecEnv([make_env(env_id=ENV_NAME, rank=i, scale_obs=True, expert_data_dir=BC_EXPERT_DATA_DIR) for i in range(env_num_used_in_callback)])

    # TODO: normalize reward!!!
    if ENV_NORMALIZE:
        vec_env = VecNormalize(venv=vec_env, norm_obs=False, norm_reward=True, gamma=GAMMA)
        eval_env = VecNormalize(venv=eval_env, norm_obs=False, norm_reward=False, gamma=GAMMA, training=False)
        # Callback will automatically synchronize normalize-related parameters between training_env and eval_env when called!!!
        eval_env_in_callback = VecNormalize(venv=eval_env_in_callback, norm_obs=False, norm_reward=False, gamma=GAMMA, training=False)

    # 2. Load model
    bc_policy_save_dir = PROJECT_ROOT_DIR / "checkpoints" / "bc" / BC_EXPERIMENT_NAME
    algo_ppo_for_kl_loss = PPOWithBCLoss.load(str((bc_policy_save_dir / BC_POLICY_FILE_NAME).absolute()))
    algo_ppo_for_kl_loss.policy.set_training_mode(False)
    algo_ppo = PPOWithBCLoss.load(
        str((bc_policy_save_dir / BC_POLICY_FILE_NAME).absolute()), 
        env=vec_env, 
        seed=SEED_FOR_LOAD_ALGO,
        custom_objects={
            "bc_trained_algo": algo_ppo_for_kl_loss,
            "learning_rate": linear_schedule(RL_LR_RATE),
            "kl_coef_with_bc": kl_decay_schedule,
            "target_kl_with_bc": None,
        },
        ent_coef=RL_ENT_COEF,
        kl_coef_with_bc=kl_decay_schedule,
        target_kl_with_bc=None,
        gamma=GAMMA,
        n_steps=N_STEPS,
        n_epochs=N_EPOCHS,
    )
    sb3_logger.info(str(algo_ppo.policy))

    # set sb3 logger
    algo_ppo.set_logger(sb3_logger)

    # evaluate
    reward, _ = evaluate_policy(algo_ppo.policy, eval_env, 1*env_num_used_in_eval)
    sb3_logger.info(f"Reward before RL: {reward}")

    # 3. Train value head
    for k, v in algo_ppo.policy.named_parameters():
        # print(k)
        if any([ x in k.split('.') for x in ['shared_net', 'policy_net', 'action_net']]):  # There's also log_std
            v.requires_grad = False
    # exit(0)
    # for k, v in algo_ppo.policy.named_parameters():
    #     print(k, v.requires_grad)
    
    start_time = time()
    # Fix KL coef during value-head training
    tmp_scheduler = algo_ppo.kl_coef_with_bc
    algo_ppo.kl_coef_with_bc = KL_DECAY_START
    # Keep LR constant during value-head training (avoid decay)
    tmp_lr_schedule = algo_ppo.lr_schedule
    algo_ppo.lr_schedule = lambda _: RL_LR_RATE
    algo_ppo.learn(total_timesteps=ACTIVATE_VALUE_HEAD_TRAIN_STEPS, log_interval=10)
    # restore LR schedule
    algo_ppo.lr_schedule = tmp_lr_schedule
    algo_ppo.kl_coef_with_bc = tmp_scheduler
    sb3_logger.info(f"training value head time: {time() - start_time}(s).")

    # Save model after value head training
    algo_ppo.save(bc_policy_save_dir / BC_POLICY_AFTER_VALUE_HEAD_TRAINED_FILE_NAME)

    # evaluate
    reward, _ = evaluate_policy(algo_ppo.policy, eval_env, 1*env_num_used_in_eval)
    sb3_logger.info(f"Reward after training value head: {reward}")

    # 4. Continue training
    for k, v in algo_ppo.policy.named_parameters():
        if any([ x in k.split('.') for x in ['shared_net', 'policy_net', 'action_net']]):
            v.requires_grad = True

    # for k, v in algo_ppo.policy.named_parameters():
    #     print(k, v.requires_grad)

    # SB3's built-in EvalCallback saves optimal policy based on highest average reward; changed to MyEvalCallback, saves optimal policy based on highest win rate
    eval_callback = EvalCallback(
        eval_env_in_callback, 
        best_model_save_path=str((PROJECT_ROOT_DIR / "checkpoints" / "rl" / RL_EXPERIMENT_NAME).absolute()),
        log_path=str((PROJECT_ROOT_DIR / "logs" / "rl" / RL_EXPERIMENT_NAME).absolute()), 
        eval_freq=EVAL_FREQ,  # How many env.step() calls before evaluating once, set to 1000 here, since VecEnv has 72 parallel environments, this is equivalent to 72*1000 steps before one evaluation
        n_eval_episodes=1*env_num_used_in_callback,  # How many trajectories to use per evaluation
        deterministic=True, 
        render=False,
    )

    # Since eval_callback is used, no need to set log_interval parameter
    algo_ppo.learn(total_timesteps=RL_TRAIN_STEPS, callback=eval_callback, log_interval=10)

    # evaluate
    reward, _ = evaluate_policy(algo_ppo.policy, eval_env, 3*env_num_used_in_eval)
    sb3_logger.info(f"Reward after RL: {reward}")

    # save model
    # rl_policy_save_dir = Path(__file__).parent / "checkpoints_sb3" / "rl" / RL_EXPERIMENT_NAME
    # algo_ppo.save(str(rl_policy_save_dir / RL_POLICY_FILE_NAME))

    return sb3_logger, eval_env

if __name__ == "__main__":
    sb3_logger, eval_env = train()

    # test best policy saved during training
    policy_save_dir = PROJECT_ROOT_DIR / "checkpoints" / "rl" / RL_EXPERIMENT_NAME
    algo_ppo = PPOWithBCLoss.load(str((policy_save_dir / "best_model").absolute()))

    mean_reward, _ = evaluate_policy(algo_ppo.policy, eval_env, 10)
    sb3_logger.info(f"Optimal policy score: {mean_reward}")
