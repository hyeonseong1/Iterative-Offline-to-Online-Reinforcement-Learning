import numpy as np
import torch as th
import h5py
import argparse
import sys 
from pathlib import Path

import gym
import d4rl
from stable_baselines3.ppo import PPO
from stable_baselines3.common.policies import BasePolicy

PROJECT_ROOT_DIR = Path(__file__).parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from models.sb3_model import PPOWithBCLoss
from utils.sb3_env_wrappers import ScaledObservationWrapper
from configs.load_config import load_config
from utils.load_data import load_data, scale_obs, split_data, load_data_from_my_cache
from load_data import load_data as load_data_from_hd5

import os

def get_reset_data():
    data = dict(
        observations = [],
        next_observations = [],
        actions = [],
        rewards = [],
        terminals = [],
        timeouts = [],
        logprobs = [],
        qpos = [],
        qvel = []
    )
    return data


def rollout(policy: BasePolicy, env: ScaledObservationWrapper, max_path: int=1000, num_data: int=1000_000, deterministic: bool=False):

    data = get_reset_data()
    traj_data = get_reset_data()

    _returns = 0
    t = 0 
    done = False
    s = env.reset()
    while len(data['rewards']) < num_data:

        # torch.from_numpy(*args, **kwargs).float().to(device)
        torch_s = th.from_numpy(np.expand_dims(s, axis=0)).float().to("cuda")
        # action, _ = policy.predict(observation=s, deterministic=True)
        actions, values, log_prob = policy.forward(obs=torch_s, deterministic=deterministic)  # When using deterministic sampling, BC performance based on dataset is poor

        a = actions.to('cpu').detach().numpy().squeeze()
        log_prob = log_prob.to('cpu').detach().numpy().squeeze()


        #mujoco only
        qpos, qvel = env.sim.data.qpos.ravel().copy(), env.sim.data.qvel.ravel().copy()

        ns, rew, done, infos = env.step(a)

        _returns += rew

        t += 1
        timeout = False
        terminal = False
        if t == max_path:
            timeout = True
        elif done:
            terminal = True

        # Need to restore s first
        origin_s = env.inverse_scale_state(s)
        origin_ns = env.inverse_scale_state(ns)

        traj_data['observations'].append(origin_s)
        traj_data['actions'].append(a)
        traj_data['next_observations'].append(origin_ns)
        traj_data['rewards'].append(rew)
        traj_data['terminals'].append(terminal)
        traj_data['timeouts'].append(timeout)
        traj_data['logprobs'].append(log_prob)
        traj_data['qpos'].append(qpos)
        traj_data['qvel'].append(qvel)

        s = ns
        if terminal or timeout:
            print('Finished trajectory. Len=%d, Returns=%f. Progress:%d/%d' % (t, _returns, len(data['rewards']), num_data))
            s = env.reset()
            t = 0
            _returns = 0
            for k in data:
                data[k].extend(traj_data[k])
            traj_data = get_reset_data()
    
    new_data = dict(
        observations=np.array(data['observations']).astype(np.float32),
        actions=np.array(data['actions']).astype(np.float32),
        next_observations=np.array(data['next_observations']).astype(np.float32),
        rewards=np.array(data['rewards']).astype(np.float32),
        terminals=np.array(data['terminals']).astype(bool),
        timeouts=np.array(data['timeouts']).astype(bool)
    )
    new_data['infos/action_log_probs'] = np.array(data['logprobs']).astype(np.float32)
    new_data['infos/qpos'] = np.array(data['qpos']).astype(np.float32)
    new_data['infos/qvel'] = np.array(data['qvel']).astype(np.float32)

    for k in new_data:
        new_data[k] = new_data[k][:num_data]
    return new_data

def save_data_to_h5_file(data: dict, h5_file_path: str):
    if not os.path.exists(h5_file_path):
        os.makedirs(os.path.dirname(h5_file_path), exist_ok=True)
    hfile = h5py.File(h5_file_path, 'w')
    for k in data:
        hfile.create_dataset(k, data=data[k], compression='gzip')

    hfile.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pass in config file")
    parser.add_argument("--config_file_name", type=str, help="Config file name", default="iter_1/seed1/kl_1/medium_halfcheetah_kl1.json")
    parser.add_argument("--deterministic", action="store_true", help="Whether to use deterministic sampling")
    args = parser.parse_args()

    custom_config = load_config(args.config_file_name)

    if args.deterministic:
        print("Deterministic sampling.")
    else:
        print("Sampling from distribution!!!!!!!!!!!!!!!!")

    ENV_NAME = custom_config["env"]["name"]
    RL_EXPERIMENT_NAME = custom_config["rl_bc"]["experiment_name"]

    # prepare env
    origin_env = gym.make(ENV_NAME)

    BC_EXPERT_DATA_DIR = custom_config["bc"].get("data_cache_dir", "cache")

    if BC_EXPERT_DATA_DIR == "cache":
        print("\033[32m load data from d4rl cache. \033[0m")
        obs, acts, infos = load_data(origin_env)
    else:
        data_file: Path = PROJECT_ROOT_DIR / BC_EXPERT_DATA_DIR
        print(f"\033[31m load data from {str(data_file.absolute())} \033[0m")
        dataset = load_data_from_hd5(str(data_file.absolute()))
        obs, acts, infos = load_data_from_my_cache(dataset)

    scaled_obs, scaler = scale_obs(obs)
    env = ScaledObservationWrapper(env=origin_env, scaler=scaler)

    # load policy
    policy_save_dir = PROJECT_ROOT_DIR / "checkpoints" / "rl" / RL_EXPERIMENT_NAME
    algo_ppo = PPOWithBCLoss.load(str((policy_save_dir / "best_model").absolute()))
    algo_ppo.policy.set_training_mode(False)

    dataset = rollout(algo_ppo.policy, env, max_path=1000, num_data=1_000_000, deterministic=args.deterministic)

    save_data_to_h5_file(dataset, str(PROJECT_ROOT_DIR / "rollout" / "data" / (RL_EXPERIMENT_NAME + ".hdf5")))