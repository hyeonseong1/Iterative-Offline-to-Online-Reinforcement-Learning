#!/bin/bash

python exp_on_d4rl/sb3_bc_train.py --config_file_name exp_on_d4rl/configs/iter_1/seed1/kl_1e-1/medium_hopper_kl1e-1.json

python exp_on_d4rl/sb3_rl_train_after_bc.py --config_file_name exp_on_d4rl/configs/iter_1/seed1/kl_1e-1/medium_hopper_kl1e-1.json

python exp_on_d4rl/rollout/rollout.py --config_file_name exp_on_d4rl/configs/iter_1/seed1/kl_1e-1/medium_hopper_kl1e-1.json