import numpy as np

def forward_euler_step(y, f, t, delta_t):
    return delta_t * f(y, t) + y


def centered_euler_step(y_curr, y_prev, f, t, delta_t):
    return 2 * delta_t * f(y_curr, t) + y_prev


if __name__ == '__main__':
    y = 0
    slope = .1
    delta_t = 1
    n = 10

    times = np.arange(0, n * delta_t, delta_t)

    for t in times:
        print(t, y)
        y = forward_euler_step(y, lambda y, t: slope, t, delta_t)
