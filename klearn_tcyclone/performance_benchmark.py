"""Functions for performance benchmarking."""

from time import perf_counter
import functools
from collections.abc import Callable


# Adapted from https://realpython.com/python-timer/#creating-a-python-timer-decorator
def timer(func: Callable) -> Callable:
    """Wrapper to time function evaluation.

    Args:
        func (Callable): Function that is timed.

    Returns:
        Callable: Wrapped function that outputs a tuple (value, elapsed_time), where
            value is the output of func.
    """

    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        tic = perf_counter()
        value = func(*args, **kwargs)
        toc = perf_counter()
        elapsed_time = toc - tic
        return value, elapsed_time

    return wrapper_timer
