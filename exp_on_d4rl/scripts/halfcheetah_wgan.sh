python sb3_bc_train.py --config_file_name iter_1/seed1/medium_halfcheetah_.json
python sb3_rl_train_after_bc_wgan.py --config_file_name iter_1/seed1/kl_1e-1/medium_halfcheetah_wgan.json
python rollout/rollout.py --config_file_name iter_1/seed1/medium_halfcheetah_.json --deterministic
python sb3_bc_train.py --config_file_name iter_2/seed1/kl_1e-1/medium_halfcheetah_wgan.json
python sb3_rl_train_after_bc_wgan.py --config_file_name iter_2/seed1/kl_1e-1/medium_halfcheetah_wgan.json

