import numpy as np
from DDE_solver.rkh_refactor import *


def f(t, y, yq):
    return -yq + 5


def phi(t):
    return -0.5 if -1 <= t <= 0 else 4.5


def alpha(t, y):
    return t - 2 - y**2


t_span = [0, 3]
discs = [-1]

solver = solve_dde(f, alpha, phi, t_span, discs=discs)
plt.plot(solver.t, solver.y, color="blue", label='aproxx')
plt.legend()
plt.show()
