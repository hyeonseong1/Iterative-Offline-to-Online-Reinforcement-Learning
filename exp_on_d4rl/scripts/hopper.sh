##### Annealing ##### <
# iter1 (hopper)
python sb3_bc_train.py --config_file_name iter_1/seed1/medium_hopper_annealing.json
python sb3_rl_train_after_bc.py --config_file_name iter_1/seed1/medium_hopper_annealing.json
python rollout/rollout.py --config_file_name iter_1/seed1/medium_hopper_annealing.json --deterministic
# iter2 (hopper)
python sb3_bc_train.py --config_file_name iter_2/seed1/medium_hopper_annealing.json
python sb3_rl_train_after_bc.py --config_file_name iter_2/seed1/medium_hopper_annealing.json
python rollout/rollout.py --config_file_name iter_2/seed1/medium_hopper_annealing.json --deterministic
# iter3 (hopper)
python sb3_bc_train.py --config_file_name iter_3/seed1/medium_hopper_annealing.json
python sb3_rl_train_after_bc.py --config_file_name iter_3/seed1/medium_hopper_annealing.json
python rollout/rollout.py --config_file_name iter_3/seed1/medium_hopper_annealing.json --deterministic
