import numpy as np
from DDE_solver.rkh_refactor import *


def f(t, y, x, z):
    return -z


def phi(t):
    return 1 - t


def phi_t(t):
    return -1


def alpha(t, y):
    return t - (1/4)*y**2


def beta(t, y):
    return t - (1/4)*y**2


def real_sol_1(t):
    return 1 + t


def real_sol_2(t):
    return 1 + t if 0 <= t <= 1 else 3 - t


t_span = [0, 5 - np.sqrt(12)]
# discs = [(-1, 1, 0)]

# solutionList = solve_dde(f, alpha, phi, t_span,
#                          neutral=True, d_phi=phi_t, beta=beta)

solutionList = solve_ndde(t_span, f, alpha, beta, phi, phi_t)

print(f'{'='*80}')
tt = np.linspace(t_span[0], t_span[1], 100)
realsol1 = np.array([real_sol_1(t) for t in tt])
realsol2 = np.array([real_sol_2(t) for t in tt])
# sol1 = [solutionList.solutions[0].eta(i) for i in tt]
# sol2 = [solutionList.solutions[1].eta(i) for i in tt]
plt.plot(tt, realsol1, color="red", label='real solution1')
plt.plot(tt, realsol2, color="green", label='real solution2')
plt.plot(solutionList.t, solutionList.y, color="orange", label='approx')
# plt.plot(tt, sol1, color="blue", label='aproxx1')
# plt.plot(tt, sol2, color="black", label='aproxx2')
plt.legend()
plt.show()
