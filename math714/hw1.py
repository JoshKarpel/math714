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


class BackwardEuler(ODESolver):
    method = 'BE'

    def solve(self):
        raise NotImplementedError


class Trapezoid(ODESolver):
    method = 'Trap'

    def solve(self):
        raise NotImplementedError


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


def plot_comparison_to_analytic(analytic, solver, **kwargs):
    fig = plt.figure(figsize = (7, 7 * 2 / 3), dpi = 600)
    fig.set_tight_layout(True)
    axis = plt.subplot(111)

    solver.attach_y_vs_t_to_axis(axis)
    line, = plt.plot(solver.t, analytic(np.linspace(solver.t[0], solver.t[-1], 10 * len(solver.t))), label = 'Analytic', linestyle = '--', color = 'orange')

    title = axis.set_title(r'y vs. t, using {} with $\Delta t = {}$'.format(solver.method, solver.delta_t), fontsize = 15)
    title.set_y(1.05)
    axis.set_xlabel(r'$t$', fontsize = 15)
    axis.set_ylabel(r'$y$', fontsize = 15)

    axis.set_xlim(solver.t[0], solver.t[-1])
    axis.grid(True, color = 'black', linestyle = ':')
    axis.tick_params(axis = 'both', which = 'major', labelsize = 10)

    axis.legend(loc = 'best', fontsize = 12)

    utils.save_current_figure(**kwargs)

    plt.close()


def plot_error(analytic, solver, **kwargs):
    fig = plt.figure(figsize = (7, 7 * 2 / 3), dpi = 600)
    fig.set_tight_layout(True)
    axis = plt.subplot(111)

    line, = plt.plot(solver.t, np.abs(analytic(solver.t) - solver.y), label = r'$\left| y_{\mathrm{analytic}}(t) - y_{\mathrm{numeric}}(t) \right|$')

    axis.set_yscale('log')

    title = axis.set_title(r'Error in y vs. t, using {} with $\Delta t = {}$'.format(solver.method, solver.delta_t), fontsize = 15)
    title.set_y(1.05)
    axis.set_xlabel(r'$t$', fontsize = 15)
    axis.set_ylabel(r'$\left| y_{\mathrm{analytic}}(t) - y_{\mathrm{numeric}}(t) \right|$', fontsize = 15)

    axis.set_xlim(solver.t[0], solver.t[-1])
    axis.grid(True, color = 'black', linestyle = ':')
    axis.tick_params(axis = 'both', which = 'major', labelsize = 10)

    # axis.legend(loc = 'best', fontsize = 12)

    utils.save_current_figure(**kwargs)

    plt.close()


def plot_final_error_vs_time_step(time_steps, solver, ode, analytic, **kwargs):
    final_errors = np.zeros(np.shape(time_steps)) * np.NaN
    for ii, dt in enumerate(time_steps):
        sol = solver(ode, delta_t = dt)

        sol.solve()

        # plot_comparison_to_analytic(analytic, sol, name = '{}_dt{}_cmp'.format(sol.method, dt), target_dir = OUT_DIR)
        # plot_error(analytic, sol, name = '{}_dt{}_error'.format(sol.method, dt), target_dir = OUT_DIR)

        final_errors[ii] = np.abs(sol.y[-1] - analytic(sol.t[-1]))

    fig = plt.figure(figsize = (7, 7 * 2 / 3), dpi = 600)
    fig.set_tight_layout(True)
    axis = plt.subplot(111)

    line, = plt.plot(time_steps, final_errors)

    axis.set_yscale('log')
    axis.set_xscale('log')

    title = axis.set_title(r'Error in $y(t=1)$ vs. time step, using {}'.format(sol.method), fontsize = 15)
    title.set_y(1.05)
    axis.set_xlabel(r'$\Delta t$', fontsize = 15)
    axis.set_ylabel(r'$\left| y_{\mathrm{analytic}}(t=1) - y_{\mathrm{numeric}}(t=1) \right|$', fontsize = 15)

    axis.set_xlim(time_steps[0], time_steps[-1])
    axis.grid(True, color = 'black', linestyle = '--')
    axis.tick_params(axis = 'both', which = 'major', labelsize = 10)

    # axis.legend(loc = 'best', fontsize = 12)

    utils.save_current_figure(**kwargs)

    plt.close()


def plot_hw1(time_steps, solvers, ode, analytic, **kwargs):
    final_errors = {solver: np.zeros(np.shape(time_steps)) * np.NaN for solver in solvers}

    for ii, dt in enumerate(time_steps):
        solutions = []

        for solver in solvers:
            sol = solver(ode, delta_t = dt)

            sol.solve()

            # plot_comparison_to_analytic(analytic, sol, name = '{}_dt{}_cmp'.format(sol.method, dt), target_dir = OUT_DIR)
            # plot_error(analytic, sol, name = '{}_dt{}_error'.format(sol.method, dt), target_dir = OUT_DIR)

            final_errors[solver][ii] = np.abs(sol.y[-1] - analytic(sol.t[-1]))
            solutions.append(sol)

        fig = plt.figure(figsize = (7, 7 * 2 / 3), dpi = 600)
        fig.set_tight_layout(True)
        axis = plt.subplot(111)

        for sol in solutions:
            sol.attach_y_vs_t_to_axis(axis)

        dense_t = np.linspace(sol.t[0], sol.t[-1], 10 * len(sol.t))
        line, = plt.plot(dense_t, analytic(dense_t), label = 'Analytic', linestyle = '--', color = 'orange')

        title = axis.set_title(r'y vs. t, with $\Delta t = {}$'.format(dt), fontsize = 15)
        title.set_y(1.05)
        axis.set_xlabel(r'$t$', fontsize = 15)
        axis.set_ylabel(r'$y$', fontsize = 15)

        axis.set_xlim(sol.t[0], sol.t[-1])
        axis.grid(True, color = 'black', linestyle = ':')
        axis.tick_params(axis = 'both', which = 'major', labelsize = 10)

        axis.legend(loc = 'best', fontsize = 12)

        utils.save_current_figure(name = 'solutions_dt{}'.format(dt), **kwargs)

        plt.close()

    fig = plt.figure(figsize = (7, 7 * 2 / 3), dpi = 600)
    fig.set_tight_layout(True)
    axis = plt.subplot(111)

    for solver in solvers:
        plt.plot(time_steps, final_errors[solver], label = solver.method)

    axis.set_yscale('log')
    axis.set_xscale('log')

    title = axis.set_title(r'Error in $y(t=1)$ vs. time step, using {}'.format(sol.method), fontsize = 15)
    title.set_y(1.05)
    axis.set_xlabel(r'$\Delta t$', fontsize = 15)
    axis.set_ylabel(r'$\left| y_{\mathrm{analytic}}(t=1) - y_{\mathrm{numeric}}(t=1) \right|$', fontsize = 15)

    axis.set_xlim(time_steps[0], time_steps[-1])
    axis.grid(True, color = 'black', linestyle = '--')
    axis.tick_params(axis = 'both', which = 'major', labelsize = 10)

    axis.legend(loc = 'best', fontsize = 12)

    utils.save_current_figure(name = 'error_comparison', **kwargs)

    plt.close()


if __name__ == '__main__':
    OUT_DIR = os.path.join(os.getcwd(), 'hw1')


    def ode_p2(y, t):
        return (100 * ((t ** 3) - y)) + (3 * (t ** 2))


    def analytic_solution_p2(t):
        return (t ** 3) + np.exp(-100 * t)


    # solver = ForwardEuler(problem2, delta_t = 0.0001)
    #
    # solver.solve()
    #
    # print(solver.t)
    # print(solver.y)
    #
    # solver.plot_y_vs_t(name = 'test', target_dir = OUT_DIR)

    # dt = [.1, .05, .01, .005, .001, .0005, .0001, .00001]
    p_min = 1
    p_max = 5
    p_pts = 50
    dt = np.around(np.logspace(-p_min, -p_max, num = p_pts), 7)

    # for delta_t in dt:
    #     rksolver = RungeKutta(problem2, delta_t = delta_t)
    #     rksolver.solve()
    #
    #     rksolver.plot_y_vs_t(name = 'rk_dt{}'.format(delta_t), target_dir = OUT_DIR)

        # plot_comparison_to_analytic(analytic, rksolver, name = 'rk_cmp_dt{}'.format(delta_t), target_dir = OUT_DIR)
        # plot_error(analytic, rksolver, name = 'rk_error_dt{}'.format(delta_t), target_dir = OUT_DIR)

    # plot_final_error_vs_time_step(dt, RungeKutta, ode_p2, analytic_solution_p2, name = 'rk_error_vs_step_{}to{}_{}pts'.format(p_min, p_max, p_pts), target_dir = OUT_DIR)
    plot_hw1(dt, [ForwardEuler, RungeKutta], ode_p2, analytic_solution_p2, target_dir = OUT_DIR)

        # forward_euler_solver = ODESolver(foo, 1, t_step = 0.00001, method = forward_euler_solver)
        #
        # forward_euler_solver.evolve()
        #
        # print(forward_euler_solver.y)
