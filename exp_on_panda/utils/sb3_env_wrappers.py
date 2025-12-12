from gymnasium import ObservationWrapper, ActionWrapper, Env, spaces
import panda_gym
from sklearn.preprocessing import StandardScaler
from typing import TypeVar
import numpy as np
from pathlib import Path
import sys
from typing import Union

# from gymnasium core.py
ObsType = TypeVar("ObsType")
WrapperObsType = TypeVar("WrapperObsType")
ActType = TypeVar("ActType")
WrapperActType = TypeVar("WrapperActType")


class ScaledObservationWrapper(ObservationWrapper):
    
    def __init__(self, env: Env, scaler: StandardScaler):
        super().__init__(env)

        # Scaling is independent of the simulator, only used in the learner
        # Observations fed to the policy network have values in [-inf, inf] range, but are standardized        
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=env.observation_space.shape)
        self.state_scalar = scaler
    
    def scale_state(self, state_var: np.ndarray) -> np.ndarray:
        """Scale the state returned by the simulator to [0, 1] range
        """
        if len(state_var.shape) == 1:
            tmp_state_var = state_var.reshape((1, -1))
            return self.state_scalar.transform(tmp_state_var).reshape((-1))
        elif len(state_var.shape) == 2:
            return self.state_scalar.transform(state_var)
        else:
            raise TypeError("state_var can only be 1D or 2D!")

    def observation(self, observation: ObsType) -> WrapperObsType:
        # Check observation type
        if type(observation) == np.ndarray:
            return self.scale_state(observation)
        else:
            return self.scale_state(np.array(observation))
    
    def inverse_scale_state(self, state_var: np.ndarray) -> np.ndarray:
        """Convert state from [0, 1] range back to the original state defined by the simulator. For testing!!!
        """
        if len(state_var.shape) == 1:
            tmp_state_var = state_var.reshape((1, -1))
            return self.state_scalar.inverse_transform(tmp_state_var).reshape((-1))
        elif len(state_var.shape) == 2:
            return self.state_scalar.inverse_transform(state_var)
        else:
            raise TypeError("state_var can only be 1D or 2D!")


class ScaledDictObservationWrapper(ObservationWrapper):
    """For environments with dict observation type, keys include observation, desired_goal, achieved_goal
    """
    def __init__(self, env: Env, obs_scaler: StandardScaler, achieved_goal_scaler: StandardScaler, desired_goal_scaler: StandardScaler):
        super().__init__(env)

        # Scaling is independent of the simulator, only used in the learner
        # Observations fed to the policy network have values in [-inf, inf] range, but are standardized        
        self.observation_space = spaces.Dict(
            dict(
                observation=spaces.Box(-np.inf, np.inf, shape=env.observation_space["observation"].shape, dtype=np.float32),
                desired_goal=spaces.Box(-np.inf, np.inf, shape=env.observation_space["desired_goal"].shape, dtype=np.float32),
                achieved_goal=spaces.Box(-np.inf, np.inf, shape=env.observation_space["achieved_goal"].shape, dtype=np.float32),
            )
        )
        self.obs_scalar: StandardScaler = obs_scaler
        self.achieved_goal_scaler: StandardScaler = achieved_goal_scaler
        self.desired_goal_scaler: StandardScaler = desired_goal_scaler

    def _scale_single_state(self, state_var: dict) -> dict:

        assert "observation" in state_var, "state_var must include key: observation!!!"
        assert "achieved_goal" in state_var, "state_var must include key: achieved_goal!!!"
        assert "desired_goal" in state_var, "state_var must include key: desired_goal!!!"
        assert type(state_var["observation"]) == np.array, "observation must be np.ndarray type!!!"
        assert type(state_var["achieved_goal"]) == np.array, "achieved_goal must be np.ndarray type!!!"
        assert type(state_var["desired_goal"]) == np.array, "desired_goal must be np.ndarray type!!!"

        tmp_obs_var = state_var["observation"].reshape((1, -1))
        tmp_a_goal_var = state_var["achieved_goal"].reshape((1, -1))
        tmp_d_goal_var = state_var["desired_goal"].reshape((1, -1))
        
        return {
            "observation": self.obs_scalar.transform(tmp_obs_var).reshape((-1)),
            "achieved_goal": self.achieved_goal_scaler.transform(tmp_a_goal_var).reshape((-1)),
            "desired_goal": self.desired_goal_scaler.transform(tmp_d_goal_var).reshape((-1)),
        }

    def scale_state(self, state_var: Union[dict, np.ndarray]) -> Union[dict, np.ndarray]:
        """Scale the state returned by the simulator to [0, 1] range
        """
        if type(state_var) == dict:
            return self._scale_single_state(state_var)
        elif type(state_var) == np.ndarray:
            np.array([self._scale_single_state(item) for item in state_var])
        else:
            raise TypeError("state_var can only be dict or np.array type!")

    def observation(self, observation: ObsType) -> WrapperObsType:
        # Check observation type
        assert type(observation) == dict, "observation can only be dict type"
        return self.scale_state(observation)
    
    def _inverse_scale_single_state(self, state_var: dict) -> dict:
        assert "observation" in state_var, "state_var must include key: observation!!!"
        assert "achieved_goal" in state_var, "state_var must include key: achieved_goal!!!"
        assert "desired_goal" in state_var, "state_var must include key: desired_goal!!!"
        assert type(state_var["observation"]) == np.array, "observation must be np.ndarray type!!!"
        assert type(state_var["achieved_goal"]) == np.array, "achieved_goal must be np.ndarray type!!!"
        assert type(state_var["desired_goal"]) == np.array, "desired_goal must be np.ndarray type!!!"

        tmp_obs_var = state_var["observation"].reshape((1, -1))
        tmp_a_goal_var = state_var["achieved_goal"].reshape((1, -1))
        tmp_d_goal_var = state_var["desired_goal"].reshape((1, -1))

        return {
            "observation": self.obs_scalar.inverse_transform(tmp_obs_var).reshape((-1)),
            "achieved_goal": self.achieved_goal_scaler.inverse_transform(tmp_a_goal_var).reshape((-1)),
            "desired_goal": self.desired_goal_scaler.inverse_transform(tmp_d_goal_var).reshape((-1)),
        }

    def inverse_scale_state(self, state_var: Union[dict, np.ndarray]) -> Union[dict, np.ndarray]:
        """Convert state from [0, 1] range back to the original state defined by the simulator. For testing!!!
        """
        if type(state_var) == dict:
            return self._inverse_scale_single_state(state_var)
        elif type(state_var) == np.ndarray:
            return np.array([self._inverse_scale_single_state(item) for item in state_var])
        else:
            raise TypeError("state_var can only be dict or np.array type!")