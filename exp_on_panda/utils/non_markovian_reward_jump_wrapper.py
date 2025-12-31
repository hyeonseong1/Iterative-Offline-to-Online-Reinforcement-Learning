import gymnasium as gym
from gymnasium import Wrapper
import numpy as np


class NMRJumpWrapper(Wrapper):
    """
    Non-Markovian Reward Jump Wrapper.
    Tracks multiple goals in sequence (jump_cnt_target goals).
    The agent needs to reach each goal in sequence.
    """
    
    def __init__(
        self,
        env: gym.Env,
        jump_cnt_target: int = 2,
        use_original_reward: bool = False,
        non_terminal_reward: float = 0.0,
        terminal_reward: float = 1.0,
    ):
        """
        Initialize the NMR Jump wrapper.
        
        Args:
            env: The environment to wrap
            jump_cnt_target: Number of goals to reach in sequence
            use_original_reward: Whether to use the original environment reward
            non_terminal_reward: Reward given for non-terminal steps
            terminal_reward: Reward given when reaching terminal state
        """
        super().__init__(env)
        self.jump_cnt_target = jump_cnt_target
        self.use_original_reward = use_original_reward
        self.non_terminal_reward = non_terminal_reward
        self.terminal_reward = terminal_reward
        
        # Track goals reached
        self.goals_reached = 0
        self.current_goal_index = 0
        
        # Store original goal getter if available
        if hasattr(self.env.unwrapped, 'task'):
            self.task = self.env.unwrapped.task
    
    def reset(self, **kwargs):
        """Reset the environment and initialize goal tracking."""
        obs, info = self.env.reset(**kwargs)
        
        # Reset goal tracking
        self.goals_reached = 0
        self.current_goal_index = 0
        
        return obs, info
    
    def step(self, action):
        """Step the environment and compute jump-based reward."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Check if current goal is reached
        if hasattr(self.task, 'get_achieved_goal'):
            achieved_goal = self.task.get_achieved_goal()
        elif isinstance(obs, dict) and 'achieved_goal' in obs:
            achieved_goal = np.array(obs['achieved_goal'])
        else:
            # Fallback: use first 3 elements of observation
            if isinstance(obs, dict) and 'observation' in obs:
                achieved_goal = np.array(obs['observation'][:3])
            else:
                achieved_goal = np.array(obs[:3])
        
        if hasattr(self.task, 'get_goal'):
            current_goal = self.task.get_goal()
        elif hasattr(self.task, 'goal'):
            current_goal = np.array(self.task.goal)
        elif isinstance(obs, dict) and 'desired_goal' in obs:
            current_goal = np.array(obs['desired_goal'])
        else:
            current_goal = None
        
        if current_goal is not None:
            distance_to_goal = np.linalg.norm(achieved_goal - current_goal)
            if hasattr(self.task, 'distance_threshold'):
                threshold = self.task.distance_threshold
            else:
                threshold = 0.01  # Default threshold
            
            if distance_to_goal < threshold:
                self.goals_reached += 1
                self.current_goal_index += 1
                
                # If we haven't reached all goals, reset to get a new goal
                if self.goals_reached < self.jump_cnt_target and not terminated and not truncated:
                    # Reset the environment to get a new goal
                    obs, info = self.env.reset()
                    # Update tracking
                    self.current_goal_index = self.goals_reached
        
        # Compute reward
        if self.use_original_reward:
            nmr_reward = reward
        else:
            if terminated or truncated:
                # Only give terminal reward if we've reached all goals
                if self.goals_reached >= self.jump_cnt_target:
                    nmr_reward = self.terminal_reward
                else:
                    nmr_reward = self.non_terminal_reward
            else:
                nmr_reward = self.non_terminal_reward
        
        return obs, nmr_reward, terminated, truncated, info

