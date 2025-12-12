##### Annealing ##### <- reward 10000
# iter1 (halfcheetah)
python sb3_bc_train.py --config_file_name iter_1/seed1/medium_halfcheetah_annealing.json
python sb3_rl_train_after_bc.py --config_file_name iter_1/seed1/medium_halfcheetah_annealing.json
python rollout/rollout.py --config_file_name iter_1/seed1/medium_halfcheetah_annealing.json --deterministic
# iter2 (halfcheetah)
python sb3_bc_train.py --config_file_name iter_2/seed1/medium_halfcheetah_annealing.json
python sb3_rl_train_after_bc.py --config_file_name iter_2/seed1/medium_halfcheetah_annealing.json
python rollout/rollout.py --config_file_name iter_2/seed1/medium_halfcheetah_annealing.json --deterministic
# iter3 (halfcheetah)
python sb3_bc_train.py --config_file_name iter_3/seed1/medium_halfcheetah_annealing.json
python sb3_rl_train_after_bc.py --config_file_name iter_3/seed1/medium_halfcheetah_annealing.json
python rollout/rollout.py --config_file_name iter_3/seed1/medium_halfcheetah_annealing.json --deterministic
