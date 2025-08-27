# from DDE_solver.rkh_state import *
# from DDE_solver.rkh_step_rejection import *
# from DDE_solver.rkh_testing import *
# from DDE_solver.rkh import *
import numpy as np
# from DDE_solver.rkh_ovl_simp_newton import *
# from DDE_solver.rkh_fast_ov_test_disc import *
from DDE_solver.rkh_vectorize import *
# from DDE_solver.rkh_NDDE import *

# WARN: STATE EXAMPLE


def f(t, y, x):
    x1, x2, x3, x4 = x
    return -x1 + x2 - x3*x4


def phi(t):
    return 1 if t < 0 else 0


def alpha(t, y):
    return t - 1


def real_sol(t):
    if 0 <= t <= 1:
        return 6*np.exp(5*t) - 1
    if 1 < t <= 2:
        return 6*(np.exp(5) + t - 6/5)*np.exp(5*t - 5) + 1/5


t_span = [0, 2]

d_f = [0, lambda t, y, x: 5, lambda t, y, x: 1]
d_alpha = [lambda t, y: 1, lambda t, y: 0]
def d_phi(t): return 0


solver = Solver(f, alpha, phi, t_span, d_f, d_alpha, d_phi)


solver.solve_dde()
tt = np.linspace(t_span[0], t_span[1], 100)
realsol = np.array([real_sol(t) for t in tt])
sol = np.array([solver.eta(i) for i in tt])
# for i in range(len(tt)):
#     print(tt[i], realsol[i] - sol[i])
print("max", max(abs(sol - realsol)))
solution = np.array([real_sol(t) for t in solver.t])
print('adnaed', max(solver.y - solution))


plt.plot(tt, realsol, color="red", label='real solution')
plt.plot(tt, sol, color="blue", label='aproxx')
plt.legend()
plt.show()
