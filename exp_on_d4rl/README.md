# Experiments on D4RL

## Prepare Python Environment

Refer to [D4RL](https://github.com/Farama-Foundation/D4RL) for setting up the RL environments, the other python packages are the same as `requirements.txt`.

**Note:** D4RL datasets are automatically downloaded when you first use them. They will be stored in `~/.d4rl/datasets` by default. You don't need to manually download them - they'll be downloaded automatically when you first access a dataset. Don't cancel before D4RL datasets download finished.

#### (Optional) Troubleshooting D4RL/mujoco_py Installation

If you encounter compilation errors with `mujoco_py` when using D4RL environments, try the following:

1. **Downgrade Cython** (most common fix):
   ```bash
   pip uninstall cython
   pip install cython==0.29.21
   ```

2. **Install system dependencies** (Ubuntu/Debian):
   ```bash
   sudo apt-get install libosmesa6-dev patchelf
   ```

3. **Install compatible GCC** (if needed):
   ```bash
   sudo apt-get install gcc-8
   ```

4. **Reinstall mujoco_py**:
   ```bash
   pip uninstall mujoco_py
   pip install mujoco_py
   ```

After fixing mujoco_py, try running your D4RL training script again.

## Prepare Demonstrations

BC training scripts will download demonstrations automatically with the help of D4RL.

### Move into directory first
```bash
cd exp_on_d4rl
```

## Test

### Pre-train policies by BC → Fine-tune policies by RL → Update demonstrations
sb3_bc_train.py → sb3_rl_train_after_bc.py → rollout.py

```bash
### Halfcheetah ###
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

### Hopper ###
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
```