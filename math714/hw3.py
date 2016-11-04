__author__ = 'Josh Karpel'

import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sps

from math714 import utils


class LinearHyperbolicSystemSolver:
    method = None

    def __init__(self, *, x_lower_bound = -1, x_upper_bound = 1, x_points = 100, u_analytic, v_analytic, matrix,
                 t_initial = 0, t_final = 0.35, delta_t = 0.01):
        self.x = np.linspace(x_lower_bound, x_upper_bound, x_points + 2)
        self.x_dense = np.linspace(x_lower_bound, x_upper_bound, x_points * 10)
        self.delta_x = self.x[1] - self.x[0]
        self.x_points = x_points

        self.u = dict()
        self.v = dict()

        self.u_analytic = u_analytic
        self.v_analytic = v_analytic

        self.matrix = matrix

        self.delta_t = delta_t
        self.time_index = 0
        self.t_initial = t_initial
        self.t_final = t_final

        self.u[0] = self.u_analytic(self.x, 0)
        self.v[0] = self.v_analytic(self.x, 0)

    def time(self, time_index):
        return self.delta_t * time_index

    def solve(self):
        raise NotImplementedError

    def plot_uv_vs_t(self, time_index = 0, **kwargs):
        fig = plt.figure(figsize = (7, 7 * 2 / 3), dpi = 600)
        fig.set_tight_layout(True)
        axis = plt.subplot(111)

        t = self.time(time_index)

        plt.plot(self.x, self.u[time_index], color = 'blue', label = '$u_{\mathrm{numeric}}$')
        plt.plot(self.x, self.u_analytic(self.x, t), color = 'teal', label = '$u_{\mathrm{analytic}}$')

        plt.plot(self.x, self.v[time_index], color = 'red', label = '$v_{\mathrm{numeric}}$')
        plt.plot(self.x, self.v_analytic(self.x, t), color = 'orange', label = '$v_{\mathrm{analytic}}$')

        axis.set_xlim(self.x[0], self.x[-1])
        axis.set_ylim(-2, 2)

        axis.set_xlabel(r'$x$', fontsize = 15)
        axis.set_ylabel(r'$u(x, t), \, v(x, t)$', fontsize = 15)
        title = axis.set_title(r'Solution at $t = {}$'.format(round(t, 3)), fontsize = 15)
        title.set_y(1.05)

        axis.grid(True, color = 'gray', linestyle = ':', alpha = 0.9)
        axis.tick_params(axis = 'both', which = 'major', labelsize = 10)
        axis.legend(loc = 'best', fontsize = 12)

        utils.save_current_figure(name = '{}_i={}_t={}'.format(self.method, time_index, round(t, 3)), **kwargs)

        plt.close()


class UpwindSolver(LinearHyperbolicSystemSolver):
    method = 'Upwind'

    def __init__(self, **kwargs):
        super(UpwindSolver, self).__init__(**kwargs)

    def solve(self):
        while self.time(self.time_index) < self.t_final:
            self.time_index += 1


class LFSolver(LinearHyperbolicSystemSolver):
    method = 'LaxFriedrichs'

    def __init__(self, **kwargs):
        super(LFSolver, self).__init__(**kwargs)

        self.r = self.delta_t / (2 * self.delta_x)

    def solve(self):
        while self.time(self.time_index) < self.t_final:
            current_u = self.u[self.time_index]
            current_v = self.v[self.time_index]

            new_u = current_u.copy()
            new_v = current_v.copy()

            new_u[1:-1] = (current_u[:-2] + current_u[2:]) / 2
            new_v[1:-1] = (current_v[:-2] + current_v[2:]) / 2

            new_u[1:-1] += -self.r * self.matrix[0, 0] * (current_u[2:] - current_u[:-2])
            new_u[1:-1] += -self.r * self.matrix[0, 1] * (current_v[2:] - current_v[:-2])

            new_v[1:-1] += -self.r * self.matrix[1, 0] * (current_u[2:] - current_u[:-2])
            new_v[1:-1] += -self.r * self.matrix[1, 1] * (current_v[2:] - current_v[:-2])

            self.u[self.time_index + 1] = new_u
            self.v[self.time_index + 1] = new_v

            self.time_index += 1


class LWSolver(LinearHyperbolicSystemSolver):
    method = 'LaxWendroff'

    def __init__(self, **kwargs):
        super(LWSolver, self).__init__(**kwargs)

        self.r = self.delta_t / (2 * self.delta_x)
        self.r2 = (self.delta_t ** 2) / (2 * (self.delta_x ** 2))

        self.matrix_squared = self.matrix.dot(self.matrix)

    def solve(self):
        while self.time(self.time_index) < self.t_final:
            current_u = self.u[self.time_index]
            current_v = self.v[self.time_index]

            new_u = current_u.copy()
            new_v = current_v.copy()

            new_u[1:-1] += -self.r * self.matrix[0, 0] * (current_u[2:] - current_u[:-2])
            new_u[1:-1] += self.r2 * self.matrix_squared[0, 0] * (current_u[2:] - 2 * current_u[1:-1] + current_u[:-2])

            new_u[1:-1] += -self.r * self.matrix[0, 1] * (current_v[2:] - current_v[:-2])
            new_u[1:-1] += self.r2 * self.matrix_squared[0, 1] * (current_v[2:] - 2 * current_v[1:-1] + current_v[:-2])

            new_v[1:-1] += -self.r * self.matrix[1, 0] * (current_u[2:] - current_u[:-2])
            new_v[1:-1] += self.r2 * self.matrix_squared[1, 0] * (current_u[2:] - 2 * current_u[1:-1] + current_u[:-2])

            new_v[1:-1] += -self.r * self.matrix[1, 1] * (current_v[2:] - current_v[:-2])
            new_v[1:-1] += self.r2 * self.matrix_squared[1, 1] * (current_v[2:] - 2 * current_v[1:-1] + current_v[:-2])

            self.u[self.time_index + 1] = new_u
            self.v[self.time_index + 1] = new_v

            self.time_index += 1


if __name__ == '__main__':
    OUT_DIR = os.path.join(os.getcwd(), 'hw3')


    def w_p(x, t):
        xi = x - (np.sqrt(2) * t)

        return np.where(np.less_equal(xi, 0), 1, -np.sqrt(2))


    def w_m(x, t):
        xi = x + (np.sqrt(2) * t)

        return np.where(np.less_equal(xi, 0), 1, np.sqrt(2))


    def u(x, t):
        return (w_p(x, t) + w_m(x, t)) / 2


    def v(x, t):
        return (w_p(x, t) - w_m(x, t)) / (2 * np.sqrt(2))


    matrix = np.array([[0, 1], [2, 0]])

    lf = LFSolver(u_analytic = u, v_analytic = v, matrix = matrix,
                  t_final = .01, delta_t = .01, x_points = 100)
    lw = LWSolver(u_analytic = u, v_analytic = v, matrix = matrix,
                  t_final = .01, delta_t = .01, x_points = 100)

    lf.solve()
    lw.solve()

    for time_index in lf.u:
        print(time_index)
        lf.plot_uv_vs_t(time_index, target_dir = OUT_DIR)
        lw.plot_uv_vs_t(time_index, target_dir = OUT_DIR)
