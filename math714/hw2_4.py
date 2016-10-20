__author__ = 'Josh Karpel'

import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sps

from math714.hw2_3 import tdma
from math714 import utils


class ReactionDiffusionSO:
    def __init__(self, points = 1000, u_init = lambda x, y: 1, epsilon = 0.05):
        self.points = points
        self.points_total = points ** 2
        self.x, self.y = np.linspace(0, 1, points), np.linspace(0, 1, points)
        self.x_mesh, self.y_mesh = np.meshgrid(self.x, self.y, indexing = 'ij')

        self.u = u_init(self.x_mesh, self.y_mesh)
        self.epsilon = epsilon

    def plot_u(self, colormesh = False, **kwargs):
        fig = plt.figure(figsize = (7, 7 * .75), dpi = 600)
        fig.set_tight_layout(True)
        axis = plt.subplot(111)

        postfix = ''

        contour = axis.contour(self.x_mesh, self.y_mesh, self.u,
                     levels = [0], colors = 'green')
        plt.clabel(contour, inline = 1, fontsize = 10)

        if colormesh:
            color = axis.pcolormesh(self.x_mesh, self.y_mesh, self.u,
                                    cmap = plt.get_cmap('gray'), shading = 'gouraud')

            cbar = plt.colorbar(color)

            postfix += 'colormesh'

        axis.set_xlabel(r'$x$', fontsize = 15)
        axis.set_ylabel(r'$y$', fontsize = 15)
        title = axis.set_title(r'$u(x,y)$', fontsize = 15)
        title.set_y(1.05)

        utils.save_current_figure('RDSO_u' + postfix, **kwargs)

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

    rdso = ReactionDiffusionSO(u_init = ellipse(a, b, alpha))

    rdso.plot_u(target_dir = OUT_DIR)
    rdso.plot_u(colormesh = True, target_dir = OUT_DIR)



