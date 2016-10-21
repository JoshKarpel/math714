__author__ = 'Josh Karpel'

import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sps

from math714.hw2_3 import tdma
from math714 import utils


class ReactionDiffusionSO:
    def __init__(self, points = 100, u_init = lambda x, y: 1, epsilon = 0.05,
                 t_initial = 0, t_final = 1, delta_t = 0.01):
        self.points = points
        self.points_total = points ** 2
        self.x, self.y = np.linspace(0, 1, points), np.linspace(0, 1, points)
        self.x_mesh, self.y_mesh = np.meshgrid(self.x, self.y, indexing = 'ij')
        self.mesh_shape = np.shape(self.x_mesh)
        self.delta_x = self.x[1] - self.x[0]

        self.u = u_init(self.x_mesh, self.y_mesh)
        self.epsilon = epsilon

        self.delta_t = delta_t
        self.t = np.arange(t_initial, t_final + 0.5 * delta_t, delta_t)
        self.time_steps = len(self.t)
        self.time_index = 0

        self.r = self.delta_t / (2 * (self.delta_x ** 2))  # lambda is a reserved name...

        self.explicit_matrix = sps.diags([1 * self.r * np.ones(self.points_total - 1),
                                          1 - 2 * self.r * np.ones(self.points_total),
                                          1 * self.r * np.ones(self.points_total)],
                                         offsets = (-1, 0, 1))
        self.implicit_upper = -1 * self.r * np.ones(self.points_total - 1)
        self.implicit_diag = 1 + 2 * self.r * np.ones(self.points_total)
        self.implicit_lower = -1 * self.r * np.ones(self.points_total - 1)

    def plot_u(self, colormesh = False, postfix = '', **kwargs):
        fig = plt.figure(figsize = (7, 7 * .75), dpi = 600)
        fig.set_tight_layout(True)
        axis = plt.subplot(111)

        contour = axis.contour(self.x_mesh, self.y_mesh, self.u,
                     levels = [0], colors = 'green')
        plt.clabel(contour, inline = 1, fontsize = 10)

        if colormesh:
            color = axis.pcolormesh(self.x_mesh, self.y_mesh, self.u,
                                    cmap = plt.get_cmap('gray'), shading = 'gouraud')

            cbar = plt.colorbar(color)

            postfix += '_colormesh'

        axis.set_xlabel(r'$x$', fontsize = 15)
        axis.set_ylabel(r'$y$', fontsize = 15)
        title = axis.set_title(r'$u(x,y, t = {})$'.format(self.t[self.time_index]), fontsize = 15)
        title.set_y(1.05)

        utils.save_current_figure('RDSO_u' + postfix, **kwargs)

        plt.close()

    def flatten_mesh(self, mesh, flatten_along):
        """Return a mesh flattened along one of the mesh coordinates ('x' or 'y')."""
        if flatten_along == 'x':
            flat = 'F'
        elif flatten_along == 'y':
            flat = 'C'
        else:
            raise ValueError("{} is not a valid specifier for flatten_along (valid specifiers: 'x', 'y')".format(flatten_along))

        return mesh.flatten(flat)

    def wrap_vector(self, vector, wrap_along):
        if wrap_along == 'x':
            wrap = 'F'
        elif wrap_along == 'y':
            wrap = 'C'
        else:
            raise ValueError("{} is not a valid specifier for wrap_vector (valid specifiers: 'x', 'y')".format(wrap_along))

        return np.reshape(vector, self.mesh_shape, wrap)

    def evolve(self):
        u_1 = self.wrap_vector(self.explicit_matrix.dot(self.flatten_mesh(self.u, 'x')), 'x')
        u_2 = tdma(self.implicit_upper, self.implicit_diag, self.implicit_lower, self.flatten_mesh(u_1, 'y'))
        u_3 = self.wrap_vector(self.explicit_matrix.dot(u_2), 'y')
        u_4 = self.wrap_vector(tdma(self.implicit_upper, self.implicit_diag, self.implicit_lower, self.flatten_mesh(u_3, 'x')), 'x')

        self.u = u_4

    def solve(self, plot_intermediate = False, **kwargs):
        while self.time_index < self.time_steps - 1:
            print(self.time_index)
            self.evolve()
            self.time_index += 1

            if plot_intermediate:
                self.plot_u(colormesh = True, postfix = '_{}'.format(self.time_index), **kwargs)


if __name__ == '__main__':
    OUT_DIR = os.path.join(os.getcwd(), 'hw2')

    def ellipse(a, b, alpha, inside = 1, outside = -1):
        def foo(x_test, y_test):
            x = x_test - 0.5
            y = y_test - 0.5
            return np.where(np.greater(((x * np.cos(alpha) + y * np.sin(alpha)) / a) ** 2 + ((x * np.sin(alpha) - y * np.cos(alpha)) / b) ** 2, 1),
                            outside, inside)

        return foo

    a = .1
    b = .2
    alpha = np.pi / 4

    rdso = ReactionDiffusionSO(u_init = ellipse(a, b, alpha), points = 200)

    rdso.plot_u(target_dir = OUT_DIR)
    rdso.plot_u(colormesh = True, target_dir = OUT_DIR)

    rdso.solve(plot_intermediate = True, target_dir = OUT_DIR)



