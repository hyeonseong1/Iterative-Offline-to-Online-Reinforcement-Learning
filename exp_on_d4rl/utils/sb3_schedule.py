from typing import Callable
import numpy as np

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func

def linear_increase_schedule(start_value: float, end_value: float) -> Callable[[float], float]:
    """
    Linear schedule that increases from start_value to end_value.
    
    :param start_value: Value at the beginning of training (progress=1.0)
    :param end_value: Value at the end of training (progress=0.0)
    :return: schedule function
    """
    def func(progress_remaining: float) -> float:
        # progress_remaining: 1.0 -> 0.0
        # done_fraction: 0.0 -> 1.0
        done_fraction = 1.0 - progress_remaining
        return start_value + done_fraction * (end_value - start_value)
    
    return func

def step_increase_schedule(
    total_steps: float,
    start_value: float,
    end_value: float,
    step_size: float = 1_000_000.0,
) -> Callable[[float], float]:
    """
    Stepwise schedule that increases from start_value to end_value every `step_size` timesteps.
    Uses SB3's progress_remaining (1.0 -> 0.0) to infer elapsed steps.
    """
    # Avoid division by zero
    total_steps = max(total_steps, 1.0)
    # Number of step bumps possible
    num_bumps = max(int(total_steps // step_size), 1)
    increment = (end_value - start_value) / num_bumps

    def func(progress_remaining: float) -> float:
        # progress_remaining: 1.0 -> 0.0
        done_steps = (1.0 - progress_remaining) * total_steps
        bumps = int(done_steps // step_size)
        val = start_value + increment * bumps
        return float(np.clip(val, min(start_value, end_value), max(start_value, end_value)))

    return func
