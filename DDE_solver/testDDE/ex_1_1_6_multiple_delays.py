import numpy as np
import matplotlib.pyplot as plt
from DDE_solver.rkh_refactor import *


def f(t, y, x):
    x1, x2, x3, x4 = x
    return -x1 + x2 - x3*x4


def phi(t):
    if t < 0:
        return 1
    else:
        return 0


def alpha(t, y):
    return [t-1,  t-2, t-3, t-4]


def real_sol(t):

    if 0 <= t <= 1:
        return -t
    elif 1 < t <= 2:
        return (1/2) * t**2 - t - (1/2)
    elif 2 < t <= 3:
        return (-1/6) * t**3 + (1/2) * t**2 - (7/6)
    elif 3 < t <= 4:
        return (1/24) * t**4 - (1/6) * t**3 - (1/4) * t**2 + t - (19/24)
    elif 4 < t <= 5:
        return (-1/120) * t**5 + (1/6) * t**4 - (5/3) * t**3 + (109/12) * t**2 - 24 * t + (2689/120)
    else:
        return np.nan


t_span = [0, 10]

discs = [(0, 1, 0)]

solver = solve_dde(f, alpha, phi, t_span, discs=discs)
tt = np.linspace(t_span[0], t_span[1], 100)
realsol = np.array([real_sol(t) for t in tt])
sol = np.array([solver.eta(i) for i in tt])
# for i in range(len(tt)):
#     print(tt[i], realsol[i] - sol[i])
print("max", np.max(abs(sol - realsol)))
solution = np.array([real_sol(t) for t in solver.t])
print('adnaed', np.max(np.squeeze(solver.y) - np.squeeze(solution)))


plt.plot(tt, realsol, color="red", label='real solution')
plt.plot(tt, sol, color="blue", label='aproxx')
plt.legend()
plt.show()
