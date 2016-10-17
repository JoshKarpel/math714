__author__ = 'Josh Karpel'

import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sps

from math714 import utils


def tdma(a_upper, a_diag, a_lower, b):
    """
    Solve a x = b for x for matrix a and vector b.

    :param a_upper: upper diagonal of matrix a
    :param a_diag: diagonal of matrix a
    :param a_lower: lower diagonal of matrix a
    :param b: result of a x
    :return: result of of a^-1 b
    """
    n = len(a_diag)

    new_upper = np.zeros(len(a_upper), dtype = np.float64)
    new_upper[0] = a_upper[0] / a_diag[0]
    for ii in range(1, n - 1):
        new_upper[ii] = a_upper[ii] / (a_diag[ii] - (a_lower[ii] * new_upper[ii - 1]))

    new_b = np.zeros(len(b), dtype = np.float64)
    new_b[0] = b[0] / a_diag[0]
    for ii in range(1, n):
        new_b[ii] = (b[ii] - (a_lower[ii - 1] * new_b[ii - 1])) / (a_diag[ii] - (a_lower[ii - 1] * new_upper[ii - 1]))

    x = np.zeros(len(b), dtype = np.float64)
    x[n - 1] = new_b[n - 1]
    for ii in reversed(range(0, n - 1)):
        x[ii] = new_b[ii] - (new_upper[ii] * x[ii + 1])

    return x


class HeatEquationCN:
    method = 'CN'

    def __init__(self, x_lower_bound = 0, x_upper_bound = 1, x_points = 100,
                 u_initial = lambda x: 0, u_analytic = lambda x, t: 0,
                 t_initial = 0, t_final = 1, delta_t = 0.01):
        self.x = np.linspace(x_lower_bound, x_upper_bound, x_points)
        self.delta_x = self.x[1] - self.x[0]
        self.x_points = len(self.x)

        self.delta_t = delta_t
        self.t = np.arange(t_initial, t_final + 0.5 * delta_t, delta_t)  # store an array of times
        self.time_steps = len(self.t)

        self.u = [np.zeros(self.x_points) for _ in self.t]
        self.u[0] = u_initial(self.x)
        self.time_index = 0

        self.u_analytic = u_analytic

        self.r = self.delta_t / (2 * (self.delta_x ** 2))

        self.explicit_matrix = sps.diags([1 * self.r * np.ones(self.x_points - 1), 1 - 2 * self.r * np.ones(self.x_points), 1 * self.r * np.ones(self.x_points)],
                                         offsets = (-1, 0, 1))
        self.implicit_upper = -1 * self.r * np.ones(self.x_points - 1)
        self.implicit_diag = 1 + 2 * self.r * np.ones(self.x_points)
        self.implicit_lower = -1 * self.r * np.ones(self.x_points - 1)

    def solve(self):
        while self.time_index < self.time_steps - 1:
            rhs = self.explicit_matrix.dot(self.u[self.time_index])
            u = tdma(self.implicit_upper, self.implicit_diag, self.implicit_lower, rhs)

            self.time_index += 1
            self.u[self.time_index] = u

    def attach_u_vs_x_to_axis(self, axis, t_index = 0):
        line, = axis.plot(self.x, self.u[t_index], label = r'$u(t={})$'.format(self.t[t_index]))

        return line

    def plot_u_vs_x(self, t_index = 0, overlay_analytic = True, **kwargs):
        fig = plt.figure(figsize = (7, 7 * 2 / 3), dpi = 600)
        fig.set_tight_layout(True)
        axis = plt.subplot(111)

        self.attach_u_vs_x_to_axis(axis, t_index = t_index)
        if overlay_analytic:
            axis.plot(self.x, self.u_analytic(self.x, self.t[t_index]), linestyle = '--', label = r'$u_{\mathrm{analytic}}$')

        title = axis.set_title(r'$u(t={})$ vs. $x$, using {} with $\Delta t = {}$'.format(self.t[t_index], self.method, self.delta_t), fontsize = 15)
        title.set_y(1.05)
        axis.set_xlabel(r'$x$', fontsize = 15)
        axis.set_ylabel(r'$u(t={})$'.format(self.t[t_index]), fontsize = 15)

        axis.set_xlim(self.t[0], self.t[-1])
        axis.grid(True, color = 'black', linestyle = ':')
        axis.tick_params(axis = 'both', which = 'major', labelsize = 10)

        if overlay_analytic:
            plt.legend(loc = 'best')

        utils.save_current_figure(**kwargs)

        plt.close()


if __name__ == '__main__':
    OUT_DIR = os.path.join(os.getcwd(), 'hw2')


    def u_init(x):
        return 10 * np.sin(np.pi * x)


    def u_analytic(x, t):
        return u_init(x) * np.exp(-(np.pi ** 2) * t)


    solver = HeatEquationCN(u_initial = u_init, u_analytic = u_analytic)
    solver.solve()

    for ii, t in enumerate(solver.t):
        solver.plot_u_vs_x(t_index = ii, name = 'sol_t={}'.format(t), target_dir = OUT_DIR)
