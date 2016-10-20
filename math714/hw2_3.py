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
        self.x = np.linspace(x_lower_bound, x_upper_bound, x_points + 2)
        # self.x = np.delete(self.x, 0)  # remove boundaries
        # self.x = np.delete(self.x, -1)

        # self.x = np.linspace(x_lower_bound, x_upper_bound, x_points)
        self.x_dense = np.linspace(x_lower_bound, x_upper_bound, x_points * 10)
        self.delta_x = self.x[1] - self.x[0]
        self.x_points = x_points

        self.delta_t = delta_t
        self.t = np.arange(t_initial, t_final + 0.5 * delta_t, delta_t)
        self.time_steps = len(self.t)

        self.u = [np.zeros(self.x_points + 2) for _ in self.t]
        self.u[0] = u_initial(self.x)
        self.time_index = 0

        self.u_analytic = u_analytic

        self.r = self.delta_t / (2 * (self.delta_x ** 2))  # lambda is a reserved name...

        self.explicit_matrix = sps.diags([1 * self.r * np.ones(self.x_points - 1),
                                          1 - 2 * self.r * np.ones(self.x_points),
                                          1 * self.r * np.ones(self.x_points)],
                                         offsets = (-1, 0, 1))
        self.implicit_upper = -1 * self.r * np.ones(self.x_points - 1)
        self.implicit_diag = 1 + 2 * self.r * np.ones(self.x_points)
        self.implicit_lower = -1 * self.r * np.ones(self.x_points - 1)

    def solve(self):
        while self.time_index < self.time_steps - 1:
            rhs = self.explicit_matrix.dot(self.u[self.time_index][1:-1])
            u = tdma(self.implicit_upper, self.implicit_diag, self.implicit_lower, rhs)

            self.time_index += 1
            self.u[self.time_index][1:-1] = u

    def l_norm(self, u = None, norm = 'inf'):
        """
        Return the l^{norm} norm.

        If u is None, uses self.u[self.time_index].

        norm should be 'inf', 1, or 2.
        """
        if u is None:
            u = self.u[self.time_index]

        if norm == 'inf':
            return np.max(np.abs(u))
        elif norm == 1:
            return self.delta_x * np.sum(np.abs(u))
        elif norm == 2:
            return np.sqrt(self.delta_x * np.sum(np.abs(u) ** 2))
        else:
            raise ValueError('Invalid norm specifier: {}'.format(norm))

    def attach_u_vs_x_to_axis(self, axis, t_index = 0):
        line, = axis.plot(self.x, self.u[t_index], label = r'$u(t={})$'.format(np.around(self.t[t_index], 3)))

        return line

    def plot_u_vs_x(self, t_index = 0, overlay_analytic = True, **kwargs):
        fig = plt.figure(figsize = (7, 7 * 2 / 3), dpi = 600)
        fig.set_tight_layout(True)
        axis = plt.subplot(111)

        self.attach_u_vs_x_to_axis(axis, t_index = t_index)
        if overlay_analytic:
            axis.plot(self.x_dense, self.u_analytic(self.x_dense, self.t[t_index]), linestyle = '--', label = r'$u_{\mathrm{analytic}}$')

        title = axis.set_title(r'$u(t={})$ vs. $x$, using {} with $\Delta t = {}$'.format(np.around(self.t[t_index], 3), self.method, self.delta_t), fontsize = 15)
        title.set_y(1.05)
        axis.set_xlabel(r'$x$', fontsize = 15)
        axis.set_ylabel(r'$u(t={})$'.format(np.around(self.t[t_index], 3)), fontsize = 15)

        axis.set_xlim(self.t[0], self.t[-1])
        axis.grid(True, color = 'black', linestyle = ':')
        axis.tick_params(axis = 'both', which = 'major', labelsize = 10)

        if overlay_analytic:
            plt.legend(loc = 'best')

        utils.save_current_figure(**kwargs)

        plt.close()

    def plot_u_vs_x_vs_t(self, t_indices = (0, 20, 40, 60, 80, 100), **kwargs):
        fig = plt.figure(figsize = (7, 7 * 2 / 3), dpi = 600)
        fig.set_tight_layout(True)
        axis = plt.subplot(111)

        for t_index in t_indices:
            self.attach_u_vs_x_to_axis(axis, t_index = t_index)

        title = axis.set_title(r'$u(t={})$ vs. $x$, using {} with $\Delta t = {}$'.format(np.around(self.t[t_index], 3), self.method, self.delta_t), fontsize = 15)
        title.set_y(1.05)
        axis.set_xlabel(r'$x$', fontsize = 15)
        axis.set_ylabel(r'$u(t={})$'.format(np.around(self.t[t_index], 3)), fontsize = 15)

        axis.set_xlim(self.t[0], self.t[-1])
        axis.grid(True, color = 'black', linestyle = ':')
        axis.tick_params(axis = 'both', which = 'major', labelsize = 10)

        plt.legend(loc = 'best', fontsize = 10)

        utils.save_current_figure(**kwargs)

        plt.close()


def l_inf_norm_vs_time_step_plot(time_steps, u_initial, u_analytic, **kwargs):
    errors = np.zeros(len(time_steps))
    for ii, dt in enumerate(time_steps):
        solver = HeatEquationCN(u_initial = u_initial, u_analytic = u_analytic, delta_t = dt, x_points = 200)
        solver.solve()
        errors[ii] = np.abs(solver.l_norm(u = solver.u[-1]) - solver.l_norm(u = solver.u_analytic(solver.x, solver.t[-1])))
        print(ii, dt, errors[ii])

    utils.xy_plot(time_steps, errors,
                  title = r'Error in $l^{\infty}$ Norm vs. Time Step',
                  x_label = r'Time Step $\Delta t$',
                  log_x = True, log_y = True, **kwargs)


def l_inf_norm_vs_delta_x_plot(mesh_points, u_initial, u_analytic, **kwargs):
    delta_x = np.zeros(len(mesh_points))
    errors = np.zeros(len(mesh_points))
    for ii, mesh_points in enumerate(mesh_points):
        solver = HeatEquationCN(u_initial = u_initial, u_analytic = u_analytic, delta_t = .01, x_points = mesh_points)
        solver.solve()
        delta_x[ii] = solver.delta_x
        errors[ii] = np.abs(solver.l_norm(u = solver.u[-1]) - solver.l_norm(u = solver.u_analytic(solver.x, solver.t[-1])))
        print(ii, mesh_points, errors[ii])

    utils.xy_plot(delta_x, errors,
                  title = r'Error in $l^{\infty}$ Norm vs. Mesh Spacing',
                  x_label = r'Mesh Spacing $\Delta x$',
                  log_x = True, log_y = True, **kwargs)


if __name__ == '__main__':
    OUT_DIR = os.path.join(os.getcwd(), 'hw2')


    def u_init(x):
        return 10 * np.sin(np.pi * x)


    def u_analytic(x, t):
        return u_init(x) * np.exp(-(np.pi ** 2) * t)


    solver = HeatEquationCN(u_initial = u_init, u_analytic = u_analytic)
    solver.solve()

    solver.plot_u_vs_x_vs_t(name = 'sol_vs_t', t_indices = (0, 5, 10, 15, 20, 40, 60, 80, 100), target_dir = OUT_DIR)

    # for ii, t in enumerate(solver.t):
    #     print(ii)
    #     solver.plot_u_vs_x(t_index = ii, name = 'sol_t={}'.format(np.around(t, 3)), target_dir = OUT_DIR)
    # for ii, t in enumerate(solver.t):
    #     solver.plot_u_vs_x(t_index = ii, name = 'sol_t={}'.format(t), target_dir = OUT_DIR)

    time_steps = np.linspace(.0001, .1, 1000)
    l_inf_norm_vs_time_step_plot(time_steps, u_initial = u_init, u_analytic = u_analytic, name = 'dt', target_dir = OUT_DIR)

    mesh_points = range(10, 1000, 5)
    l_inf_norm_vs_delta_x_plot(mesh_points, u_initial = u_init, u_analytic = u_analytic, name = 'dx', target_dir = OUT_DIR)

