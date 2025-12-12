#!/bin/bash

python exp_on_d4rl/sb3_rl_train.py --config_file_name iter_1/seed1/medium_halfcheetah_.json
python exp_on_d4rl/sb3_rl_train.py --config_file_name iter_1/seed2/medium_halfcheetah_.json
python exp_on_d4rl/sb3_rl_train.py --config_file_name iter_1/seed3/medium_halfcheetah_.json
python exp_on_d4rl/sb3_rl_train.py --config_file_name iter_1/seed4/medium_halfcheetah_.json
python exp_on_d4rl/sb3_rl_train.py --config_file_name iter_1/seed5/medium_halfcheetah_.json