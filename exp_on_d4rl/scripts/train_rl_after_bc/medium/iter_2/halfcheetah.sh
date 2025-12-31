#!/bin/bash

python exp_on_d4rl/sb3_rl_train_after_bc.py --config_file_name iter_2/seed1/kl_1/medium_halfcheetah_kl1.json
python exp_on_d4rl/sb3_rl_train_after_bc.py --config_file_name iter_2/seed2/kl_1/medium_halfcheetah_kl1.json
python exp_on_d4rl/sb3_rl_train_after_bc.py --config_file_name iter_2/seed3/kl_1/medium_halfcheetah_kl1.json
python exp_on_d4rl/sb3_rl_train_after_bc.py --config_file_name iter_2/seed4/kl_1/medium_halfcheetah_kl1.json
python exp_on_d4rl/sb3_rl_train_after_bc.py --config_file_name iter_2/seed5/kl_1/medium_halfcheetah_kl1.json