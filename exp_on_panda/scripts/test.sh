python sb3_bc_train.py --config_file_name configs/iter_1/seed1/reacher_256_256_kl_1e-1.json
python sb3_rl_train_after_bc.py --config_file_name configs/iter_1/seed1/reacher_256_256_kl_1e-1.json
python rollout/rollout_my_reach_by_rl_bc.py --config_file_name configs/iter_1/seed1/reacher_256_256_kl_1e-1.json

python sb3_bc_train.py --config_file_name configs/iter_2/seed1/reacher_256_256_kl_1e-1.json
python sb3_rl_train_after_bc.py --config_file_name configs/iter_2/seed1/reacher_256_256_kl_1e-1.json
python rollout/rollout_my_reach_by_rl_bc.py --config_file_name configs/iter_2/seed1/reacher_256_256_kl_1e-1.json

python sb3_bc_train.py --config_file_name configs/iter_3/seed1/reacher_256_256_kl_1e-1.json
python sb3_rl_train_after_bc.py --config_file_name configs/iter_3/seed1/reacher_256_256_kl_1e-1.json