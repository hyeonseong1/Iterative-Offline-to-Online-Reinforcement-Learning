import gymnasium as gym
from gymnasium import Wrapper
import numpy as np


class NMRWayPointWrapper(Wrapper):
    """
    Non-Markovian Reward Waypoint Wrapper.
    Creates an intermediate waypoint between the start position and the goal.
    The agent first needs to reach the waypoint, then the final goal.
    """
    
    def __init__(
        self,
        env: gym.Env,
        target_waypoint_delta: np.ndarray = np.array([0.0, 0.0, 0.1]),
        use_original_reward: bool = False,
        non_terminal_reward: float = 0.0,
        terminal_reward: float = 1.0,
    ):
        """
        Initialize the NMR Waypoint wrapper.
        
        Args:
            env: The environment to wrap
            target_waypoint_delta: Offset from goal to create waypoint (waypoint = goal + delta)
            use_original_reward: Whether to use the original environment reward
            non_terminal_reward: Reward given for non-terminal steps
            terminal_reward: Reward given when reaching terminal state
        """
        super().__init__(env)
        self.target_waypoint_delta = np.array(target_waypoint_delta)
        self.use_original_reward = use_original_reward
        self.non_terminal_reward = non_terminal_reward
        self.terminal_reward = terminal_reward
        
        # Waypoint state
        self.waypoint = None
        self.reach_way_point = False
        
        # Store original goal getter if available
        if hasattr(self.env.unwrapped, 'task'):
            self.task = self.env.unwrapped.task
    
    def reset(self, **kwargs):
        """Reset the environment and set up waypoint."""
        obs, info = self.env.reset(**kwargs)
        
        # Get the goal from the environment
        if hasattr(self.task, 'get_goal'):
            goal = self.task.get_goal()
        elif hasattr(self.task, 'goal'):
            goal = np.array(self.task.goal)
        else:
            # Try to get from observation if it's a dict
            if isinstance(obs, dict) and 'desired_goal' in obs:
                goal = np.array(obs['desired_goal'])
            else:
                raise AttributeError("Cannot find goal in environment. Ensure task has get_goal() or goal attribute.")
        
        # Calculate waypoint: waypoint = goal + target_waypoint_delta
        self.waypoint = goal + self.target_waypoint_delta
        self.reach_way_point = False
        
        return obs, info
    
    def step(self, action):
        """Step the environment and compute waypoint-based reward."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Check if waypoint is reached
        if not self.reach_way_point:
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
            
            # Check distance to waypoint
            distance_to_waypoint = np.linalg.norm(achieved_goal - self.waypoint)
            if hasattr(self.task, 'distance_threshold'):
                threshold = self.task.distance_threshold
            else:
                threshold = 0.01  # Default threshold
            
            if distance_to_waypoint < threshold:
                self.reach_way_point = True
        
        # Compute reward
        if self.use_original_reward:
            nmr_reward = reward
        else:
            if terminated or truncated:
                nmr_reward = self.terminal_reward
            else:
                nmr_reward = self.non_terminal_reward
        
        return obs, nmr_reward, terminated, truncated, info
    
    def get_goal(self):
        """Get the current goal (waypoint if not reached, final goal if reached)."""
        if hasattr(self.task, 'get_goal'):
            return self.task.get_goal()
        elif hasattr(self.task, 'goal'):
            return np.array(self.task.goal)
        else:
            return None

