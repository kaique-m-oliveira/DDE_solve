""" This is an experimental feature, still needs more development"""

import numpy as np

from DDE_solver.rkh_refactor import *


def f(t, y, yq):
    return yq + 1/2


def phi(t):
    return 1 if t <= -1 else 0


def alpha(t, y):
    return t - abs(y) - 1


def real_sol_1(t):
    return 3*t/2


def real_sol_2(t):
    return t/2


t_span = [0, 2]
discs = [(-1, 1, 0)]

solutionList = solve_dde(t_span, f, alpha, phi, discs=discs)


print(f'{'='*80}')
tt = np.linspace(t_span[0], t_span[1], 100)
realsol1 = np.array([real_sol_1(t) for t in tt])
realsol2 = np.array([real_sol_2(t) for t in tt])
sol1 = [solutionList.solutions[0].eta(i) for i in tt]
sol2 = [solutionList.solutions[1].eta(i) for i in tt]
plt.plot(tt, realsol1, color="red", label='real solution1')
plt.plot(tt, realsol2, color="green", label='real solution2')
plt.plot(tt, sol1, color="blue", label='aproxx1')
plt.plot(tt, sol2, color="black", label='aproxx2')
plt.legend()
plt.show()
