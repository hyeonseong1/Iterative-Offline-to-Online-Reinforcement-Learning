# Iterative Offline-to-Online (IO2)
This repository is improved version of "Iterative Regularized Policy Optimization with Imperfect Demonstrations" (ICML2024).

## Prepare python environment

```bash
# Create conda environment
conda create --name o2o python=3.8
conda activate o2o
pip install -r requirements.txt

# Install D4RL
cd d4rl 
pip install -e .
cd ..

# Add follow commands to the bottom of ~/.bashrc 
# Download mujoco210 binary from https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
# Unzip it to ~/.mujoco/mujoco210
nano ~/.bashrc
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/user/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
# control + O to save, control + X to exit
source ~/.bashrc

# Install Panda-Gym
pip install panda-gym

# Install FlyCraft
pip install -e fly-craft
```

## Experiments

* Refer to `exp_on_d4rl/` for experiments on HalfCheetah and Hopper.
* Refer to `exp_on_panda/` for experiments on Reach.
