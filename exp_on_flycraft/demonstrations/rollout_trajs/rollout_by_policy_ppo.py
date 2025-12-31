from pathlib import Path
import pandas as pd
from tqdm import tqdm
import itertools
import logging
import os
import sys
import argparse

from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.base_class import BaseAlgorithm
# from stable_baselines3.sac import SAC

from flycraft.env import FlyCraftEnv
from flycraft.utils_common.load_config import load_config

PROJECT_ROOT_DIR: Path = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from utils_my.sb3.my_wrappers import ScaledObservationWrapper, ScaledActionWrapper
from utils_my.models.ppo_with_bc_loss import PPOWithBCLoss
from utils_my.models.simba_policy import SimBaActorCriticPolicy


def rollout(
        algo: BaseAlgorithm,
        env_config_file: Path,
        debug: bool = False,
        trajectory_save_prefix: str = "traj",
        cur_expert_data_dir: Path = PROJECT_ROOT_DIR / "data" / "10hz_10_5_5_iter_1",
        deterministic: bool = True
):
    """
    Exhaustive search through all goals in res.csv (matching original SAC implementation).
    Args:
        algo: The policy to use for rollout
        env_config_file: Path to environment config
        debug: Debug mode
        trajectory_save_prefix: Prefix for saved trajectory files
        cur_expert_data_dir: Directory containing res.csv and where to save trajectories
        deterministic: Use deterministic policy
    """

    helper_env: FlyCraftEnv = FlyCraftEnv(config_file=env_config_file)
    scaled_obs_env = ScaledObservationWrapper(helper_env)
    scaled_act_env = ScaledActionWrapper(scaled_obs_env)
    env_config = load_config(env_config_file)

    algo.policy.set_training_mode(False)

    # Load res.csv for exhaustive search
    res_file = cur_expert_data_dir / "res.csv"
    if not res_file.exists():
        logging.error(f"res.csv not found at {res_file}")
        return

    res_df = pd.read_csv(res_file)

    # Track updates
    traj_renew_cnt = 0
    traj_add_cnt = 0

    for index, target in tqdm(res_df.iterrows(), total=res_df.shape[0]):
        # Set fixed goal from res.csv
        helper_env.task.goal_sampler.use_fixed_goal = True
        helper_env.task.goal_sampler.goal_v = target["v"]
        helper_env.task.goal_sampler.goal_mu = target["mu"]
        helper_env.task.goal_sampler.goal_chi = target["chi"]
        helper_env.task.goal_sampler.goal_expert_length = target["length"]

        traj = {
            "time": [],
            "s_phi": [],
            "s_theta": [],
            "s_psi": [],
            "s_v": [],
            "s_mu": [],
            "s_chi": [],
            "s_p": [],
            "s_h": [],
            "a_p": [],
            "a_nz": [],
            "a_pla": [],
            "a_rud": []
        }

        # Rollout
        obs, info = scaled_act_env.reset()
        terminate = False
        s_index = 0
        while not terminate:
            action, _ = algo.predict(observation=obs, deterministic=deterministic)
            obs, reward, terminate, truncated, info = scaled_act_env.step(action=action)

            traj["time"].append(s_index * 1. / env_config["task"].get("step_frequence", 10))
            traj["s_phi"].append(info["plane_state"]["phi"])
            traj["s_theta"].append(info["plane_state"]["theta"])
            traj["s_psi"].append(info["plane_state"]["psi"])
            traj["s_v"].append(info["plane_state"]["v"])
            traj["s_mu"].append(info["plane_state"]["mu"])
            traj["s_chi"].append(info["plane_state"]["chi"])
            traj["s_p"].append(info["plane_state"]["p"])
            traj["s_h"].append(info["plane_state"]["h"])
            traj["a_p"].append(info["action"]["p"])
            traj["a_nz"].append(info["action"]["nz"])
            traj["a_pla"].append(info["action"]["pla"])
            traj["a_rud"].append(info["action"]["rud"])

            s_index += 1

        # Save if successful and better than previous
        if info["is_success"]:
            prev_length = (res_df.length[(res_df.v == helper_env.task.goal_sampler.goal_v) & 
                                        (res_df.mu == helper_env.task.goal_sampler.goal_mu) & 
                                        (res_df.chi == helper_env.task.goal_sampler.goal_chi)]).iloc[0]
            
            if prev_length == 0 or s_index < prev_length:
                traj_df = pd.DataFrame(data=traj, columns=["time", "s_phi", "s_theta", "s_psi", "s_v", "s_mu", "s_chi", "s_p", "s_h", "a_p", "a_nz", "a_pla", "a_rud"])
                traj_df.to_csv(cur_expert_data_dir / f"{trajectory_save_prefix}_{int(helper_env.task.goal_sampler.goal_v)}_{int(helper_env.task.goal_sampler.goal_mu)}_{int(helper_env.task.goal_sampler.goal_chi)}.csv", index=False)
                
                res_df.loc[(res_df.v == helper_env.task.goal_sampler.goal_v) & 
                             (res_df.mu == helper_env.task.goal_sampler.goal_mu) & 
                             (res_df.chi == helper_env.task.goal_sampler.goal_chi), "length"] = s_index
                
                print(f"\033[33m更新{helper_env.task.goal_sampler.goal_v}, {helper_env.task.goal_sampler.goal_mu}, {helper_env.task.goal_sampler.goal_chi}, length: from {prev_length} to {s_index}!!!\033[0m")
                
                if prev_length == 0:
                    traj_add_cnt += 1
                else:
                    traj_renew_cnt += 1
                print(f"新增了{traj_add_cnt}条轨迹，更新了{traj_renew_cnt}条轨迹")

    res_df.to_csv(res_file, index=False)
    print(f"一共新增了{traj_add_cnt}条轨迹，更新了{traj_renew_cnt}条轨迹.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--policy-ckpt-dir", type=str, help="policy checkpoints dir", required=True)
    parser.add_argument("--env-config-dir", type=str, help="environment config dir", required=True)
    parser.add_argument("--demos-dir", type=str, help="demonstration dir", required=True)
    parser.add_argument("--policy-file-name", type=str, help="policy file name (e.g. best_model.zip)",
                        default="best_model.zip")
    # parser.add_argument("--num-trajs", type=int, help="Number of trajectories to attempt", default=28375)
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic policy")
    args = parser.parse_args()

    # Setup basic logging
    logging.basicConfig(level=logging.INFO)

    algo_save_dir = Path(os.getcwd()) / args.policy_ckpt_dir
    # Check if directory or file
    if algo_save_dir.is_dir():
        algo_save_path = algo_save_dir / args.policy_file_name
    else:
        # If user passed a file path
        algo_save_path = algo_save_dir

    env_config_file = Path(os.getcwd()) / args.env_config_dir
    cur_demonstration_dir = Path(os.getcwd()) / args.demos_dir

    print(f"Loading env from: {env_config_file}")
    env = FlyCraftEnv(config_file=env_config_file)

    print(f"Loading model from: {algo_save_path}")

    ppo_algo = PPOWithBCLoss.load(
        algo_save_path,
        env=env,  # Raw env to satisfy structure check
        policy=SimBaActorCriticPolicy,
        custom_objects={
            "observation_space": env.observation_space,
            "action_space": env.action_space
        }
    )

    rollout(
        algo=ppo_algo,
        env_config_file=env_config_file,
        # num_trajs=args.num_trajs,
        cur_expert_data_dir=cur_demonstration_dir,
        deterministic=args.deterministic
    )
