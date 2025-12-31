import gymnasium as gym
from gymnasium import Wrapper
import numpy as np
from collections import deque


class NMRWrapper(Wrapper):
    """
    Non-Markovian Reward Wrapper.
    Provides rewards based on a history window of length nmr_length.
    """
    
    def __init__(
        self,
        env: gym.Env,
        nmr_length: int = 10,
        use_original_reward: bool = False,
        non_terminal_reward: float = 0.0,
        terminal_reward: float = 1.0,
    ):
        """
        Initialize the NMR wrapper.
        
        Args:
            env: The environment to wrap
            nmr_length: Length of the history window for non-Markovian rewards
            use_original_reward: Whether to use the original environment reward
            non_terminal_reward: Reward given for non-terminal steps
            terminal_reward: Reward given when reaching terminal state
        """
        super().__init__(env)
        self.nmr_length = nmr_length
        self.use_original_reward = use_original_reward
        self.non_terminal_reward = non_terminal_reward
        self.terminal_reward = terminal_reward
        
        # History buffer for tracking states/rewards
        self.reward_history = deque(maxlen=nmr_length)
        self.terminal_history = deque(maxlen=nmr_length)
    
    def reset(self, **kwargs):
        """Reset the environment and clear history."""
        obs, info = self.env.reset(**kwargs)
        self.reward_history.clear()
        self.terminal_history.clear()
        return obs, info
    
    def step(self, action):
        """Step the environment and compute non-Markovian reward."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Store in history
        self.reward_history.append(reward)
        self.terminal_history.append(terminated or truncated)
        
        # Compute non-Markovian reward
        if self.use_original_reward:
            nmr_reward = reward
        else:
            if terminated or truncated:
                nmr_reward = self.terminal_reward
            else:
                nmr_reward = self.non_terminal_reward
        
        # Add history-based component if needed
        # For now, use simple reward structure
        return obs, nmr_reward, terminated, truncated, info

