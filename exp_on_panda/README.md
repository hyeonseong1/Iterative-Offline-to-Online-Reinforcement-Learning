# Experiments on Panda-Gym

## Prepare Python Environment

Install [Panda-Gym](https://github.com/qgallouedec/panda-gym)

```bash
pip install panda-gym
```

### Move into directory first
```bash
cd exp_on_panda
```

## Prepare Demonstrations

Rollout demonstrations by a [PID controller](https://panda-gym.readthedocs.io/en/latest/usage/manual_control.html), refer to the Jupyter script `rollout/rollout_my_reach_by_pid.ipynb`.

## Test

### Pre-train policies by BC → Fine-tune policies by RL → Update demonstrations
sb3_bc_train.py → sb3_rl_train_after_bc.py → rollout/rollout_my_reach_by_rl_bc.py

```bash
### Reach ###
    # iter1
        python sb3_bc_train.py --config_file_name configs/iter_1/seed1/reacher_annealing.json
        python sb3_rl_train_after_bc.py --config_file_name configs/iter_1/seed1/reacher_annealing.json
        python rollout/rollout_my_reach_by_rl_bc.py --config_file_name configs/iter_1/seed1/reacher_annealing.json

    # iter2
        python sb3_bc_train.py --config_file_name configs/iter_2/seed1/reacher_annealing.json
        python sb3_rl_train_after_bc.py --config_file_name configs/iter_2/seed1/reacher_annealing.json
        python rollout/rollout_my_reach_by_rl_bc.py --config_file_name configs/iter_2/seed1/reacher_annealing.json

    # iter3
        python sb3_bc_train.py --config_file_name configs/iter_3/seed1/reacher_annealing.json
        python sb3_rl_train_after_bc.py --config_file_name configs/iter_3/seed1/reacher_annealing.json
        python rollout/rollout_my_reach_by_rl_bc.py --config_file_name configs/iter_3/seed1/reacher_annealing.json
```