from imitation.data.types import TransitionsMinimal, Transitions
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
from time import time
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple
import sys
from collections import OrderedDict


PROJECT_ROOT_DIR = Path(__file__).absolute().parent.parent


def load_data_into_dict_obs(data_file: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler, StandardScaler, StandardScaler]:
    dataset_df = pd.read_csv(data_file)
    obs = dataset_df.iloc[:, 0:6]
    achieved_goals = dataset_df.iloc[:, 0:3]
    desired_goals = dataset_df.iloc[:, 6:9]
    actions = dataset_df.iloc[:, 9:12]

    obs_scaler = StandardScaler()
    achieved_goal_scaler = StandardScaler()
    desired_goal_scaler = StandardScaler()

    scaled_obs = obs_scaler.fit_transform(obs)
    scaled_achieved_goals = achieved_goal_scaler.fit_transform(achieved_goals)
    scaled_desired_goals = desired_goal_scaler.fit_transform(desired_goals)

    tmp = []

    for tmp_obs, tmp_a_goal, tmp_d_goal in zip(scaled_obs, scaled_achieved_goals, scaled_desired_goals):
        tmp_obs_dict = {
            "observation": tmp_obs,
            "achieved_goal": tmp_a_goal,
            "desired_goal": tmp_d_goal,
        }
        tmp.append(tmp_obs_dict)
    
    return np.array(tmp), actions.to_numpy(), np.array([None] * len(dataset_df)), obs_scaler, achieved_goal_scaler, desired_goal_scaler

def load_data(data_file: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    dataset_df = pd.read_csv(data_file)
    # order: achieved_goal, desired_goal, obs
    obs = dataset_df.loc[:, ('s_x', 's_y', 's_z', 's_g_x', 's_g_y', 's_g_z', 's_x', 's_y', 's_z', 's_v_x', 's_v_y', 's_v_z')].to_numpy()
    actions = dataset_df.iloc[:, 9:12].to_numpy()

    obs_scaler = StandardScaler()
    scaled_obs = obs_scaler.fit_transform(obs)

    # Compute next_obs by shifting observations
    # For the last observation, use itself as next_obs (or we could remove it)
    next_obs = np.vstack([scaled_obs[1:], scaled_obs[-1:]])
    
    # Since we don't have done information in CSV, set all to False
    # Alternatively, if there's a way to infer dones from the data, we could do that
    dones = np.zeros(len(scaled_obs), dtype=bool)

    return scaled_obs, actions, np.array([None] * len(dataset_df)), next_obs, dones, obs_scaler


def split_data(
        obs: np.ndarray, 
        acts: np.ndarray, 
        infos: np.ndarray,
        next_obs: np.ndarray,
        dones: np.ndarray,
        train_size: float=0.9, 
        validation_size: float=0.05, 
        test_size: float=0.05,
        shuffle: bool=True,
    ) -> Tuple[Transitions, Transitions, Transitions]:

    # Split into training set, validation set, and test set
    train_data, tmp_data, train_labels, tmp_labels, train_infos, tmp_infos, train_next_obs, tmp_next_obs, train_dones, tmp_dones = train_test_split(
        obs, acts, infos, next_obs, dones,
        train_size=train_size, 
        test_size=validation_size + test_size, 
        shuffle=shuffle,
        random_state=0,  # Ensure the same results every time
    )

    validation_data, test_data, validation_labels, test_labels, validation_infos, test_infos, validation_next_obs, test_next_obs, validation_dones, test_dones = train_test_split(
        tmp_data, tmp_labels, tmp_infos, tmp_next_obs, tmp_dones,
        train_size=validation_size/(validation_size + test_size),
        test_size=test_size/(validation_size + test_size),
        shuffle=shuffle,
        random_state=0,
    )

    return (
        Transitions(obs=train_data, acts=train_labels, infos=train_infos, next_obs=train_next_obs, dones=train_dones),
        Transitions(obs=validation_data, acts=validation_labels, infos=validation_infos, next_obs=validation_next_obs, dones=validation_dones),
        Transitions(obs=test_data, acts=test_labels, infos=test_infos, next_obs=test_next_obs, dones=test_dones)
    )


if __name__ == "__main__":
    pass