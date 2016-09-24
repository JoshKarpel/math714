import functools
import multiprocessing as mp
import datetime as dt
from copy import deepcopy


def multi_map(function, targets, processes = None):
    """Map a function over a list of inputs using multiprocessing."""
    if processes is None:
        processes = mp.cpu_count() - 1

    with mp.Pool(processes = processes) as pool:
        output = pool.map(function, targets)

    return output


def memoize(copy_output = False):
    """
    Returns a decorator that memoizes the result of a function call.

    :param copy_output: if True, the output of the memo will be deepcopied before returning. Defaults to False.
    :return: a Memoize decorator
    """

    class Memoize:
        def __init__(self, func):
            self.func = func
            self.memo = {}

            self.__doc__ = self.func.__doc__

        def __call__(self, *args, **kwargs):
            key = args
            for k, v in kwargs.items():
                try:
                    key += (k, tuple(v))
                except TypeError:
                    key += (k, v)

            try:
                value = self.memo[key]
            except KeyError:
                value = self.func(*args, **kwargs)
                self.memo[key] = value

            if copy_output:
                try:
                    value = value.copy()
                except AttributeError:
                    value = deepcopy(value)

            return value

        def __get__(self, obj, objtype):
            # support instance methods
            return functools.partial(self.__call__, obj)

    return Memoize


class Timer:
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.elapsed_time = None

    def __enter__(self):
        self.start_time = dt.datetime.now()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = dt.datetime.now()
        self.elapsed_time = self.end_time - self.start_time

    def __str__(self):
        if self.end_time is None:
            return 'Timer started at {}, still running, elapsed time {}'.format(self.start_time, dt.datetime.now() - self.start_time)
        else:
            return 'Timer started at {}, ended at {}, elapsed time {}'.format(self.start_time, self.end_time, self.elapsed_time)
