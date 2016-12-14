__author__ = 'Josh Karpel'

import os

import matplotlib.pyplot as plt
import numpy as np

from math714 import utils


class LevelSetSolver:
    def __init__(self, space_points = 100, delta_t = 0.0001):
        self.delta_t = delta_t
        self.t = 0
        self.t_counter = 0

        self.x, self.y = np.linspace(-0.5, 0.5, space_points + 4), np.linspace(-0.5, 0.5, space_points + 4)
        self.x_mesh, self.y_mesh = np.meshgrid(self.x, self.y, indexing = 'ij')
        self.delta_x, self.delta_y = self.x[1] - self.x[0], self.y[1] - self.y[0]

        self.u = np.zeros(np.shape(self.x_mesh))

    def plot(self, **kwargs):
        plt.figure(figsize = (7, 7), dpi = 600)

        con = plt.contour(self.x_mesh, self.y_mesh, self.u,
                          levels = [1]
                          )

        plt.title(r'$u(x,y)$ at $t = {}$'.format(np.around(self.t, 7)), fontsize = 20)
        plt.xlabel(r'$x$', fontsize = 16)
        plt.ylabel(r'$y$', fontsize = 16)

        plt.grid(True)

        utils.save_current_figure(name = 'u_{}'.format(self.t_counter), **kwargs)

        plt.close()

    def grad_x(self, mesh):
        return (mesh[2:, 1:-1] - mesh[:-2, 1:-1]) / (2 * self.delta_x)

    def grad_y(self, mesh):
        return (mesh[1:-1, 2:] - mesh[1:-1, :-2]) / (2 * self.delta_y)

    def div(self, mesh_x, mesh_y):
        return self.grad_x(mesh_x) + self.grad_y(mesh_y)

    def evolve(self):
        grad_u_x, grad_u_y = self.grad_x(self.u), self.grad_y(self.u)
        mag_grad_u = np.sqrt((grad_u_x ** 2) + (grad_u_y ** 2))
        kappa = self.div(grad_u_x / mag_grad_u, grad_u_y / mag_grad_u)

        delta_t = 0.25 * self.delta_x / np.max(np.abs(kappa))
        if self.delta_t >= delta_t:
            self.delta_t = delta_t

        self.u[2:-2, 2:-2] += self.delta_t * kappa * mag_grad_u[1:-1, 1:-1]

        self.t += self.delta_t
        self.t_counter += 1


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
    pts = 200

    OUT_DIR = os.path.join(os.getcwd(), 'hw5', str(pts))

    ls = LevelSetSolver(space_points = pts, delta_t = 0.0001)

    for ii, x in enumerate(ls.x):
        for jj, y in enumerate(ls.y):
            print(ii, jj)
            if inside(x, y, gamma):
                ls.u[ii, jj] = 1 - (distance(x, y, gamma) ** 2)
            else:
                ls.u[ii, jj] = 1 + (distance(x, y, gamma) ** 2)

    print(np.min(ls.u))
    print(np.max(ls.u))

    ls.plot(target_dir = OUT_DIR)

    for tt in range(500):
        print(tt)
        ls.evolve()
        if ls.t_counter % 10 == 0:
            ls.plot(target_dir = OUT_DIR)
