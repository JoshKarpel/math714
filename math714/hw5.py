__author__ = 'Josh Karpel'

import os

import matplotlib.pyplot as plt
import numpy as np

from math714 import utils


class LevelSetSolver:
    def __init__(self, kappa, space_points = 100, delta_t = 0.01):
        self.kappa = kappa
        self.delta_t = delta_t
        self.t = 0

        self.x, self.y = np.linspace(-0.5, 0.5, space_points), np.linspace(-0.5, 0.5, space_points)
        self.x_mesh, self.y_mesh = np.meshgrid(self.x, self.y, indexing = 'ij')

        self.u = np.zeros(np.shape(self.x_mesh))

    def plot(self, **kwargs):
        plt.figure(figsize = (7, 7), dpi = 600)

        con = plt.contour(self.x_mesh, self.y_mesh, self.u,
                          levels = [1]
                          )
        # plt.clabel(con, inline = 1, fontsize = 10)

        # pc = plt.pcolormesh(self.x_mesh, self.y_mesh, self.u, shading = 'gouraud')
        # cbar = plt.colorbar(pc)

        plt.title(r'$u(x,y)$ at $t = {}$'.format(self.t))

        utils.save_current_figure(name = 'u_{}'.format(self.t), **kwargs)

    def curvature(self):
        raise NotImplementedError


def gamma(s):
    pre = 0.1 + 0.065 * np.sin(7 * 2 * np.pi * s)
    return pre * np.cos(2 * np.pi * s), pre * np.sin(2 * np.pi * s)


def distance(x, y, f, s_tests = 200):
    distances = []
    for s in np.linspace(0, 1, s_tests):
        curve_x, curve_y = f(s)
        distances.append(np.sqrt((x - curve_x) ** 2 + (y - curve_y) ** 2))
    return min(distances)


def inside(x, y, f):
    s = np.arctan2(y, x) / (2 * np.pi)
    f_x, f_y = f(s)
    if x ** 2 + y ** 2 < f_x ** 2 + f_y ** 2:
        return True
    return False


if __name__ == '__main__':
    OUT_DIR = os.path.join(os.getcwd(), 'hw5')

    ls = LevelSetSolver(5, space_points = 100)

    for ii, x in enumerate(ls.x):
        for jj, y in enumerate(ls.y):
            # print(ii, jj, x, y, inside(x, y, gamma))
            if inside(x, y, gamma):
                ls.u[ii, jj] = 1 - 100 * (distance(x, y, gamma) ** 2)
            else:
                ls.u[ii, jj] = 1 + 100 * (distance(x, y, gamma) ** 2)

    print(np.min(ls.u))

    ls.plot(target_dir = OUT_DIR)
