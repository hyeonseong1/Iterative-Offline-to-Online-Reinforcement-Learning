import argparse
import sys
from pathlib import Path
from stable_baselines3.common.evaluation import evaluate_policy

PROJECT_ROOT_DIR = Path(__file__).parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from utils_my.models.ppo_with_bc_loss import PPOWithBCLoss
from utils_my.models.simba_policy import SimBaActorCriticPolicy
from utils_my.sb3.vec_env_helper import get_vec_env
from utils_my.sb3.my_evaluate_policy import evaluate_policy_with_success_rate
from flycraft.utils_common.load_config import load_config


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained Simba PPO model")
    parser.add_argument("--config-file-name", type=str, required=True,
                        help="Path to the training configuration file (e.g., configs/train/iteration_1/annealing/seed_1.json)")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to the trained model file (e.g., checkpoints/iter1/simba_annealing/seed1/best_model.zip)")
    parser.add_argument("--eval-episodes", type=int, default=100, help="Number of episodes to evaluate")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--deterministic", action="store_true", default=True, help="Use deterministic actions")

    args = parser.parse_args()

    train_config_path = PROJECT_ROOT_DIR / args.config_file_name
    model_path = PROJECT_ROOT_DIR / args.model_path

    print(f"Loading training config from: {train_config_path}")
    print(f"Loading model from: {model_path}")

    # Load training config to get the correct env config file
    train_config = load_config(train_config_path)
    env_config_file = train_config["env"].get("config_file", "env_config_for_sac.json")
    env_config_path = PROJECT_ROOT_DIR / "configs" / "env" / env_config_file

    print(f"Loading env config from: {env_config_path}")

    # Initialize Environment
    # Using get_vec_env to ensure same wrappers (ScaledObservationWrapper, ScaledActionWrapper) as training
    env = get_vec_env(
        num_process=1,
        seed=args.seed,
        config_file=env_config_path
    )

    # Load Model
    # Explicitly passing policy=SimBaActorCriticPolicy to handle custom policy loading
    model = PPOWithBCLoss.load(
        model_path,
        env=env,
        policy=SimBaActorCriticPolicy,
        custom_objects={
            "observation_space": env.observation_space,
            "action_space": env.action_space
        }
    )

    print(f"Evaluating for {args.eval_episodes} episodes...")

    mean_reward, std_reward, success_rate = evaluate_policy_with_success_rate(
        model=model,
        env=env,
        n_eval_episodes=args.eval_episodes,
        deterministic=args.deterministic
    )

    print("-" * 50)
    print(f"Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"Success Rate: {success_rate * 100:.2f}%")
    print("-" * 50)


if __name__ == "__main__":
    main()
