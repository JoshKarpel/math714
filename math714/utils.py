__author__ = 'Josh Karpel'

import datetime as dt
import functools
import multiprocessing as mp
import os
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

np.set_printoptions(precision = 3, linewidth = 200)


def ensure_dir_exists(path):
    """Ensure that the directory tree to the path exists."""
    split_path = os.path.splitext(path)
    if split_path[0] != path:  # path is file
        make_path = os.path.dirname(split_path[0])
    else:  # path is dir
        make_path = split_path[0]
    os.makedirs(make_path, exist_ok = True)


def save_current_figure(name = 'img', target_dir = None, img_format = 'png', scale_factor = 1):
    """Save the current matplotlib figure with the given name to the given folder."""
    if target_dir is None:
        target_dir = os.getcwd()
    path = os.path.join(target_dir, '{}.{}'.format(name, img_format))

    ensure_dir_exists(path)

    plt.savefig(path, dpi = scale_factor * plt.gcf().dpi, bbox_inches = 'tight')


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


def xy_plot(x, *y, legends = None,
            title = None, x_label = None, y_label = None,
            x_center = 0, x_range = None,
            y_lower_lim = None, y_upper_lim = None,
            log_x = False, log_y = False,
            **kwargs):
    fig = plt.figure(figsize = (7, 7 * 2 / 3), dpi = 600)
    fig.set_tight_layout(True)
    axis = plt.subplot(111)

    # plot y vs. x data
    for ii, yy in enumerate(y):
        if legends is not None:
            plt.plot(x, yy, label = legends[ii])
        else:
            plt.plot(x, yy)

    # set title
    if title is not None:
        title = axis.set_title(r'{}'.format(title), fontsize = 15)
        title.set_y(1.05)

    # set x label
    if x_label is not None:
        axis.set_xlabel(r'{}'.format(x_label), fontsize = 15)

    # set y label
    if y_label is not None:
        axis.set_ylabel(r'{}'.format(y_label), fontsize = 15)

    # set x axis limits
    if x_range is None:
        lower_limit_x = np.min(x)
        upper_limit_x = np.max(x)
    else:
        lower_limit_x = (x_center - x_range)
        upper_limit_x = (x_center + x_range)

    axis.set_xlim(lower_limit_x, upper_limit_x)

    if y_lower_lim is not None and y_upper_lim is not None:
        axis.set_ylim(y_lower_lim, y_upper_lim)

    # set whether axes are log scale
    if log_x:
        axis.set_xscale('log')
    if log_y:
        axis.set_yscale('log')

    # grid and tick options
    axis.grid(True, color = 'gray', linestyle = ':', alpha = 0.9)
    axis.tick_params(axis = 'both', which = 'major', labelsize = 10)

    # draw legend
    if legends is not None:
        axis.legend(loc = 'best', fontsize = 12)

    save_current_figure(**kwargs)

    plt.close()
