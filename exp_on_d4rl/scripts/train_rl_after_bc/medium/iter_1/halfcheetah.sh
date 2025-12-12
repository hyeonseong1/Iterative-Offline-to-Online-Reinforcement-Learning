#!/bin/bash

python exp_on_d4rl/sb3_rl_train_after_bc.py --config_file_name iter_1/seed1/kl_1e-1/medium_halfcheetah_kl1e-1.json
python exp_on_d4rl/sb3_rl_train_after_bc.py --config_file_name iter_1/seed2/kl_1e-1/medium_halfcheetah_kl1e-1.json
python exp_on_d4rl/sb3_rl_train_after_bc.py --config_file_name iter_1/seed3/kl_1e-1/medium_halfcheetah_kl1e-1.json
python exp_on_d4rl/sb3_rl_train_after_bc.py --config_file_name iter_1/seed4/kl_1e-1/medium_halfcheetah_kl1e-1.json
python exp_on_d4rl/sb3_rl_train_after_bc.py --config_file_name iter_1/seed5/kl_1e-1/medium_halfcheetah_kl1e-1.json