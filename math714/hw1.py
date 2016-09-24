__author__ = 'Josh Karpel'

import os
import numpy as np
import matplotlib.pyplot as plt

from math714 import utils


class ODESolver:
    method = None

    def __init__(self, dy_dt, y_initial = 1, t_initial = 0, t_final = 1, delta_t = 0.01):
        self.dy_dt = dy_dt

        self.delta_t = delta_t
        self.t = np.arange(t_initial, t_final + 0.5 * delta_t, delta_t)  # store an array of times
        self.time_steps = len(self.t)

        self.y = np.zeros(np.shape(self.t)) * np.NaN  # store an array of NaNs, to be replaced by a y value for each time

        self.y[0] = y_initial
        self.time_index = 0

    def solve(self):
        raise NotImplementedError

    def attach_y_vs_t_to_axis(self, axis):
        line, = axis.plot(self.t, self.y, label = '{}'.format(self.method))

        return line

    def plot_y_vs_t(self, **kwargs):
        fig = plt.figure(figsize = (7, 7 * 2 / 3), dpi = 600)
        fig.set_tight_layout(True)
        axis = plt.subplot(111)

        self.attach_y_vs_t_to_axis(axis)

        title = axis.set_title(r'y vs. t, using {} with $\Delta t = {}$'.format(self.method, self.delta_t), fontsize = 15)
        title.set_y(1.05)
        axis.set_xlabel(r't', fontsize = 15)
        axis.set_ylabel(r'y', fontsize = 15)

        axis.set_xlim(self.t[0], self.t[-1])
        axis.grid(True, color = 'black', linestyle = ':')
        axis.tick_params(axis = 'both', which = 'major', labelsize = 10)

        # axis.legend(loc = 'best', fontsize = 12)

        utils.save_current_figure(**kwargs)

        plt.close()


class ForwardEuler(ODESolver):
    method = 'FE'

    def solve(self):
        while self.time_index < self.time_steps - 1:
            self.y[self.time_index + 1] = self.y[self.time_index] + self.delta_t * self.dy_dt(self.y[self.time_index], self.t[self.time_index])

            self.time_index += 1


class RungeKutta(ODESolver):
    method = 'RK4'

    def solve(self):
        while self.time_index < self.time_steps - 1:
            k_1 = self.dy_dt(self.y[self.time_index], self.t[self.time_index])
            k_2 = self.dy_dt(self.y[self.time_index] + (self.delta_t * k_1 / 2), self.t[self.time_index] + self.delta_t / 2)
            k_3 = self.dy_dt(self.y[self.time_index] + (self.delta_t * k_2 / 2), self.t[self.time_index] + self.delta_t / 2)
            k_4 = self.dy_dt(self.y[self.time_index] + self.delta_t * k_3, self.t[self.time_index] + self.delta_t)

            self.y[self.time_index + 1] = self.y[self.time_index] + self.delta_t * (k_1 + (2 * k_2) + (2 * k_3) + k_4) / 6

            self.time_index += 1


if __name__ == '__main__':
    OUT_DIR = os.path.join(os.getcwd(), 'hw1')

    def foo(y, t):
        return 1

    def problem2(y, t):
        return (100 * ((t ** 3) - y)) + (3 * (t ** 2))

    solver = ForwardEuler(problem2, delta_t = 0.0001)

    solver.solve()

    print(solver.t)
    print(solver.y)

    solver.plot_y_vs_t(name = 'test', target_dir = OUT_DIR)

    rksolver = RungeKutta(problem2, delta_t = .01)
    rksolver.solve()

    rksolver.plot_y_vs_t(name = 'rk', target_dir = OUT_DIR)

    # forward_euler_solver = ODESolver(foo, 1, t_step = 0.00001, method = forward_euler_solver)
    #
    # forward_euler_solver.evolve()
    #
    # print(forward_euler_solver.y)