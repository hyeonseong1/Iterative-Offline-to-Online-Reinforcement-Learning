from typing import Callable

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

def linear_decay_schedule(start_value: float, end_value: float) -> Callable[[float], float]:
    """
    Linear decay schedule from start_value to end_value.
    
    :param start_value: Value at the beginning of training (progress = 1.0)
    :param end_value: Value at the end of training (progress = 0.0)
    :return: schedule function
    """
    def func(progress_remaining: float) -> float:
        # progress_remaining goes from 1.0 down to 0.0
        return end_value + progress_remaining * (start_value - end_value)
    
    return func