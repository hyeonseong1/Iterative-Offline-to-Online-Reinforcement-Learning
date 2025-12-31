## Generate Demonstrations

### Generate demonstrations with PID controller

Sample from $ V \times U \times X = [v_{min}:v_{max}:v_{interval}] \times [\mu_{min}:\mu_{max}:\mu_{interval}] \times [\chi_{min}:\chi_{max}:\chi_{interval}] $ with PID controller and save sampled trajectories in demonstrations/data/{step-frequence}hz\_{$v_{interval}$}\_{$\mu_{interval}$}\_{$\chi_{interval}$}\_{data-dir-suffix}.

```bash
# Move into flycraft directory
cd exp_on_flycraft
```

```bash
# sample trajectories by single-processing
python demonstrations/rollout_trajs/rollout_by_pid.py --data-dir-suffix v0 --step-frequence 10 --v-min 100 --v-max 300 --v-interval 10 --mu-min -85 --mu-max 85 --mu-interval 5 --chi-min -170 --chi-max 170 --chi-interval 5
```

## Training
```bash
#### Prepare demonstrations
# sample trajectories by multi-processing (base on Ray, recommended)
python demonstrations/rollout_trajs/rollout_by_pid_parallel.py --data-dir-suffix v1 --step-frequence 10 --v-min 100 --v-max 300 --v-interval 10 --mu-min -85 --mu-max 85 --mu-interval 5 --chi-min -170 --chi-max 170 --chi-interval 5

# Augment demonstrations
python demonstrations/utils/augment_trajs.py --demos-dir demonstrations/data/10hz_10_5_5_v1

# Generate cache for demonstrations
python demonstrations/utils/load_dataset.py --data-dir demonstrations/data/10hz_10_5_5_v1 --cache-dir demonstrations/cache/10hz_10_5_5_iter_1_aug


#### iter 1
# 1. Train BC (Simba)
python train_scripts/train_with_bc_simba.py --config-file-name configs/train/iteration_1/annealing/seed_1.json                                         

# 2. Train RL (Simba PPO)
python train_scripts/train_with_rl_bc_simba.py --config-file-name configs/train/iteration_1/annealing/seed_1.json

# 3. Create Data Directory for Iter 2 (Copy v1 -> v2)
cp -r demonstrations/data/10hz_10_5_5_v1 demonstrations/data/10hz_10_5_5_v2

# 4. Rollout & Update Demonstrations (using Iter 1 policy to improve v2)
python demonstrations/rollout_trajs/rollout_by_policy_ppo.py \
    --policy-ckpt-dir checkpoints/rl/iter_1/10hz_annealing_2e8steps_1/best_model \
    --env-config-dir configs/env/env_config_for_ppo.json \
    --demos-dir demonstrations/data/10hz_10_5_5_v2

# 5. Augment demonstrations
python demonstrations/utils/augment_trajs.py --demos-dir demonstrations/data/10hz_10_5_5_v2

# 6. Generate Cache for Iter 2 (v2 -> iter_2_aug)
python demonstrations/utils/load_dataset.py \
    --data-dir demonstrations/data/10hz_10_5_5_v2 \
    --cache-dir demonstrations/cache/10hz_10_5_5_iter_2_aug


#### iter 2
# 1. Train BC
python train_scripts/train_with_bc_simba.py --config-file-name configs/train/iteration_2/annealing/seed_1.json

# 2. Train RL
python train_scripts/train_with_rl_bc_simba.py --config-file-name configs/train/iteration_2/annealing/seed_1.json

# 3. Create Data Directory for Iter 3 (Copy v2 -> v3)
cp -r demonstrations/data/10hz_10_5_5_v2 demonstrations/data/10hz_10_5_5_v3

# 4. Rollout & Update Demonstrations (using Iter 2 policy to improve v3)
python demonstrations/rollout_trajs/rollout_by_policy_ppo.py \
    --policy-ckpt-dir checkpoints/rl/iter_2/10hz_annealing_2e8steps_1/best_model \
    --env-config-dir configs/env/env_config_for_ppo.json \
    --demos-dir demonstrations/data/10hz_10_5_5_v3

# 5. Augment demonstrations
python demonstrations/utils/augment_trajs.py --demos-dir demonstrations/data/10hz_10_5_5_v3

# 6. Generate Cache for Iter 3 (v3 -> iter_3_aug)
python demonstrations/utils/load_dataset.py \
    --data-dir demonstrations/data/10hz_10_5_5_v3 \
    --cache-dir demonstrations/cache/10hz_10_5_5_iter_3_aug


#### iter 3
# 1. Train BC
python train_scripts/train_with_bc_simba.py --config-file-name configs/train/iteration_3/annealing/seed_1.json

# 2. Train RL
python train_scripts/train_with_rl_bc_simba.py --config-file-name configs/train/iteration_3/annealing/seed_1.json

# 3. Create Data Directory for Iter 4 (Copy v3 -> v4)
cp -r demonstrations/data/10hz_10_5_5_v3 demonstrations/data/10hz_10_5_5_v4

# 4. Rollout & Update Demonstrations
python demonstrations/rollout_trajs/rollout_by_policy_and_update_demostrations.py \
    --policy-ckpt-dir checkpoints/iter3/simba_annealing/seed1 \
    --env-config-dir configs/env/env_config_for_ppo.json \
    --demos-dir demonstrations/data/10hz_10_5_5_v4

# 5. Generate Cache for Iter 4
python demonstrations/utils/load_dataset.py \
    --data-dir demonstrations/data/10hz_10_5_5_v4 \
    --cache-dir demonstrations/cache/10hz_10_5_5_iter_4_aug
```

## How to evaluate?
```bash
python evaluate/evaluate.py --config-file-name configs/train/iteration_3/annealing/seed_1.json --model-path checkpoints/rl/iter_3/10hz_annealing_2e8steps_1/best_model.zip
```