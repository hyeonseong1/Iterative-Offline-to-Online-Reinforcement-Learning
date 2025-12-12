import gym
import d4rl
import numpy as np
from pathlib import Path
import logging
import torch as th
import argparse
from copy import deepcopy
import sys

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure, Logger
from stable_baselines3.common.callbacks import CheckpointCallback
from imitation.algorithms import bc
from imitation.util.logger import HierarchicalLogger
from imitation.util import util
from imitation.data.types import TransitionsMinimal
from imitation.data import types as data_types
import torch.utils.data as th_data

PROJECT_ROOT_DIR = Path(__file__).parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from models.sb3_model import PPOWithBCLoss
from models.simba_policy import SimBaActorCriticPolicy
from utils.sb3_env_wrappers import ScaledObservationWrapper
from configs.load_config import load_config
from utils.sb3_schedule import linear_schedule
from utils.load_data import load_data, load_data_from_my_cache, scale_obs, split_data
from rollout.load_data import load_data as load_data_from_hd5

parser = argparse.ArgumentParser(description="Pass in config file")
parser.add_argument("--config_file_name", type=str, help="Config file name", default="iter_1/seed1/medium_halfcheetah_.json")
args = parser.parse_args()

custom_config = load_config(args.config_file_name)

ENV_NAME = custom_config["env"]["name"]

EXPERIMENT_NAME = custom_config["bc"]["experiment_name"]
SEED = custom_config["bc"]["seed"]
POLICY_FILE_SAVE_NAME = custom_config["bc"]["policy_file_save_name"]
TRAIN_EPOCHS = custom_config["bc"]["train_epochs"]
BC_BATCH_SIZE = custom_config["bc"]["batch_size"]
BC_L2_WEIGHT = custom_config["bc"].get("l2_weight", 0.0)
BC_ENT_WEIGHT = custom_config["bc"].get("ent_weight", 1e-2)
EXPERT_DATA_CACHE_DIR = custom_config["bc"]["data_cache_dir"]
USE_LOSS_CALLBACK = custom_config["bc"]["use_loss_callback"]
PROB_TRUE_ACT_THRESHOLD = custom_config["bc"]["prob_true_act_threshold"]  # During validation, if prob_true_act is greater than this value, the optimal policy for prob_true_act is saved.
LOSS_THRESHOLD = custom_config["bc"]["loss_threshold"]
DATASET_SPLIT = custom_config["bc"].get("dataset_split", [0.96, 0.02, 0.02])
BC_LR_RATE = custom_config["bc"].get("lr", 1e-4)

RL_SEED = custom_config["rl"]["seed"]
NET_ARCH = custom_config["rl_bc"]["net_arch"]
PPO_BATCH_SIZE = custom_config["rl_bc"]["batch_size"]
GAMMA = custom_config["rl_bc"]["gamma"]
KL_WITH_BC_MODEL_COEF = custom_config["rl_bc"]["kl_with_bc_model_coef"]

# Create a collate function for TransitionsMinimal (without next_obs and dones)
def transitions_minimal_collate_fn(
    batch: list,
) -> dict:
    """Custom collate_fn for TransitionsMinimal (without next_obs and dones)."""
    result = {}
    result["infos"] = [sample["infos"] for sample in batch]
    result["obs"] = data_types.stack_maybe_dictobs([sample["obs"] for sample in batch])
    result["acts"] = th_data.dataloader.default_collate([sample["acts"] for sample in batch])
    return result

# Monkey-patch make_data_loader to use the correct collate function for TransitionsMinimal
from imitation.algorithms import base as algo_base
_original_make_data_loader = algo_base.make_data_loader

def patched_make_data_loader(
    transitions, batch_size, data_loader_kwargs=None
):
    """Patched version that uses correct collate_fn for TransitionsMinimal."""
    if isinstance(transitions, TransitionsMinimal):
        if len(transitions) < batch_size:
            raise ValueError(
                f"Number of transitions in `demonstrations` {len(transitions)} "
                f"is smaller than batch size {batch_size}.",
            )
        kwargs = {
            "shuffle": True,
            "drop_last": True,
            **(data_loader_kwargs or {}),
        }
        return th_data.DataLoader(
            transitions,
            batch_size=batch_size,
            collate_fn=transitions_minimal_collate_fn,  # Use our custom collate function
            **kwargs,
        )
    else:
        return _original_make_data_loader(transitions, batch_size, data_loader_kwargs)

algo_base.make_data_loader = patched_make_data_loader

def get_ppo_algo(env):
    policy_kwargs = dict(
        full_std=True,  # Use state-dependent exploration
        # squash_output=True,  # Use state-dependent exploration
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
        bc_trained_algo=None,
        kl_coef_with_bc=KL_WITH_BC_MODEL_COEF,
        seed=SEED,
        batch_size=PPO_BATCH_SIZE,
        gamma=GAMMA,
        n_steps=256,  # Number of steps sampled per environment during rollout
        n_epochs=5,  # Number of times the sampled data is reused during training
        policy_kwargs=policy_kwargs,
        use_sde=True,  # Use state-dependent exploration
        normalize_advantage=True,
        learning_rate=linear_schedule(BC_LR_RATE),
    )


# Strategy for saving policy, two options: (1) save by highest prob_true_act; (2) save by lowest loss.
# Since bc.train()'s on_batch_end is a callback without parameters, use a closure to record the best prob_true_act via an external variable.
def on_best_act_prob_save(algo: PPO, validation_transitions: TransitionsMinimal, sb3_logger: Logger):
    best_prob = PROB_TRUE_ACT_THRESHOLD  # An initial estimate to avoid spending too much time saving models at the beginning of training
    def calc_func():
        algo.policy.set_training_mode(mode=False)
        
        nonlocal best_prob
        
        obs = util.safe_to_tensor(validation_transitions.obs).to("cuda")
        acts = util.safe_to_tensor(validation_transitions.acts).to("cuda")
        _, log_prob, entropy = algo.policy.evaluate_actions(obs, acts)
        prob_true_act = th.exp(log_prob).mean()
        if prob_true_act > best_prob:
            sb3_logger.info(f"update prob true act from {best_prob} to {prob_true_act}!")
            best_prob = prob_true_act

            # save policy
            checkpoint_save_dir = PROJECT_ROOT_DIR / "checkpoints" / "bc" / EXPERIMENT_NAME
            checkpoint_save_dir.mkdir(parents=True, exist_ok=True)

            algo.save(str(checkpoint_save_dir / POLICY_FILE_SAVE_NAME))

        algo.policy.set_training_mode(mode=True)
    return calc_func

def on_best_loss_save(algo: PPO, validation_transitions: TransitionsMinimal, loss_calculator: bc.BehaviorCloningLossCalculator, sb3_logger: Logger):
    min_loss = LOSS_THRESHOLD  # An initial estimate to avoid spending too much time saving models at the beginning of training
    def calc_func():
        algo.policy.set_training_mode(mode=False)
        
        nonlocal min_loss
        
        obs = util.safe_to_tensor(validation_transitions.obs).to("cuda")
        acts = util.safe_to_tensor(validation_transitions.acts).to("cuda")
        
        metrics: bc.BCTrainingMetrics = loss_calculator(policy=algo.policy, obs=obs, acts=acts)
        cur_loss = metrics.loss
        if cur_loss < min_loss:
            sb3_logger.info(f"update loss from {min_loss} to {cur_loss}!")
            min_loss = cur_loss

            # save policy
            checkpoint_save_dir = PROJECT_ROOT_DIR / "checkpoints" / "bc" / EXPERIMENT_NAME
            checkpoint_save_dir.mkdir(parents=True, exist_ok=True)

            algo.save(str(checkpoint_save_dir / POLICY_FILE_SAVE_NAME))

        algo.policy.set_training_mode(mode=True)
    return calc_func


def train():

    sb3_logger: Logger = configure(folder=str((PROJECT_ROOT_DIR / "logs" / "bc" / EXPERIMENT_NAME).absolute()), format_strings=['stdout', 'log', 'csv', 'tensorboard'])

    origin_env = gym.make(ENV_NAME)
    
    if EXPERT_DATA_CACHE_DIR == "cache":
        sb3_logger.info("load data from d4rl cache.")
        obs, acts, infos = load_data(origin_env)
    else:
        data_file: Path = PROJECT_ROOT_DIR / EXPERT_DATA_CACHE_DIR
        sb3_logger.info(f"load data from {str(data_file.absolute())}")
        dataset = load_data_from_hd5(str(data_file.absolute()))
        obs, acts, infos = load_data_from_my_cache(dataset)

    scaled_obs, scaler = scale_obs(obs)

    env = ScaledObservationWrapper(env=origin_env, scaler=scaler)

    algo_ppo = get_ppo_algo(env)
    sb3_logger.info(str(algo_ppo.policy))
    # print(algo_ppo.policy)

    rng = np.random.default_rng(SEED)
    
    
    train_transitions, validation_transitions, test_transitions = split_data(
        obs=scaled_obs, 
        acts=acts,
        infos=infos,
        train_size=DATASET_SPLIT[0],
        validation_size=DATASET_SPLIT[1],
        test_size=DATASET_SPLIT[2],
        shuffle=True,
    )

    sb3_logger.info(f"train_set: obs size, {train_transitions.obs.shape}, act size, {train_transitions.acts.shape}")
    sb3_logger.info(f"validation_set: obs size, {validation_transitions.obs.shape}, act size, {validation_transitions.acts.shape}")
    sb3_logger.info(f"test_set: obs size, {test_transitions.obs.shape}, act size, {test_transitions.acts.shape}")

    # print(f"policy observation space: {algo_ppo.policy.observation_space}")
    # print(f"env observation space: {env.observation_space}")

    bc_trainer = bc.BC(
        observation_space=algo_ppo.policy.observation_space,
        action_space=algo_ppo.policy.action_space,
        policy=algo_ppo.policy,
        batch_size=BC_BATCH_SIZE,
        ent_weight=BC_ENT_WEIGHT,
        l2_weight=BC_L2_WEIGHT,
        demonstrations=train_transitions,
        rng=rng,
        custom_logger=HierarchicalLogger(sb3_logger)
    )

    # train
    bc_trainer.train(
        n_epochs=TRAIN_EPOCHS,
        # on_batch_end=on_best_act_prob_save(algo_ppo, validation_transitions, sb3_logger),
        on_batch_end=on_best_loss_save(algo_ppo, validation_transitions, bc_trainer.loss_calculator, sb3_logger) if USE_LOSS_CALLBACK else on_best_act_prob_save(algo_ppo, validation_transitions, sb3_logger),
    )

    # evaluate with environment
    reward, _ = evaluate_policy(algo_ppo.policy, env, n_eval_episodes=10)
    sb3_logger.info(f"Reward after BC: {reward}, normalized: {origin_env.get_normalized_score(reward)}")

    # Final policy prob_true_act / loss on the test set
    if USE_LOSS_CALLBACK:
        test_on_loss(algo_ppo.policy, test_transitions, bc_trainer.loss_calculator, sb3_logger, "Policy at the end of training", "test set")
    else:
        test_on_prob_true_act(algo_ppo.policy, test_transitions, sb3_logger, "Policy at the end of training", "test set")

    return sb3_logger, validation_transitions, test_transitions, bc_trainer, origin_env, env


def test_on_prob_true_act(
        policy: SimBaActorCriticPolicy, 
        test_transitions: TransitionsMinimal, 
        sb3_logger: Logger, 
        policy_descreption: str, dataset_descreption: str
    ):
    policy.set_training_mode(mode=False)

    obs = util.safe_to_tensor(test_transitions.obs).to("cuda")
    acts = util.safe_to_tensor(test_transitions.acts).to("cuda")
    _, log_prob, entropy = policy.evaluate_actions(obs, acts)
    prob_true_act = th.exp(log_prob).mean()
    sb3_logger.info(f"{policy_descreption} prob_true_act on {dataset_descreption}: {prob_true_act}.")


def test_on_loss(
        policy: SimBaActorCriticPolicy, 
        test_transitions: TransitionsMinimal, 
        loss_calculator: bc.BehaviorCloningLossCalculator, 
        sb3_logger: Logger, 
        policy_descreption: str, dataset_descreption: str
    ):
    policy.set_training_mode(mode=False)

    obs = util.safe_to_tensor(test_transitions.obs).to("cuda")
    acts = util.safe_to_tensor(test_transitions.acts).to("cuda")
    
    metrics: bc.BCTrainingMetrics = loss_calculator(policy=policy, obs=obs, acts=acts)
    sb3_logger.info(f"{policy_descreption} loss on {dataset_descreption}: {metrics.loss}.")


if __name__ == "__main__":
    sb3_logger, validation_transitions, test_transitions, bc_trainer, origin_env, env = train()

    policy_save_dir = PROJECT_ROOT_DIR / "checkpoints" / "bc" / EXPERIMENT_NAME
    algo_ppo = PPOWithBCLoss.load(str((policy_save_dir / POLICY_FILE_SAVE_NAME).absolute()))

    if USE_LOSS_CALLBACK:
        test_on_loss(algo_ppo.policy, validation_transitions, bc_trainer.loss_calculator, sb3_logger, "Best policy", "validation set")
        test_on_loss(algo_ppo.policy, test_transitions, bc_trainer.loss_calculator, sb3_logger, "Best policy", "test set")
    else:
        test_on_prob_true_act(algo_ppo.policy, validation_transitions, sb3_logger, "Best policy", "validation set")
        test_on_prob_true_act(algo_ppo.policy, test_transitions, sb3_logger, "Best policy", "test set")
    
    reward, _ = evaluate_policy(algo_ppo.policy, env, n_eval_episodes=10)
    sb3_logger.info(f"Best policy score: {reward}, normalized score: {origin_env.get_normalized_score(reward)}")