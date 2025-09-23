# from DDE_solver.rkh_state import *
# from DDE_solver.rkh_step_rejection import *
# from DDE_solver.rkh_testing import *
# from DDE_solver.rkh import *
import numpy as np
# from DDE_solver.rkh_vectorize import *
# from DDE_solver.rkh_multiple_delays import *
from DDE_solver.rkh_refactor import *
# from DDE_solver.rkh_refactor_before_chatgpt import *
# from DDE_solver.rkh_state_complete import *
# from DDE_solver.rkh_NDDE import *

# WARN: STATE EXAMPLE


def f(t, y, x, z):
    q = 3/np.pi
    C = np.cos(t) - q*np.cos(q*t) + np.sin(t)
    return -y + q*z + C


def phi(t):
    return np.sin(t)


def phi_t(t):
    return np.cos(t)


def alpha(t, y):
    return (3/np.pi)*t


def real_sol(t):
    if t <= 1:
        return phi(t)
    if 1 <= t <= 10:
        return np.sin(t)


t_span = [1, 10]


solver = solve_dde(f, alpha, phi, t_span, beta=alpha,
                   neutral=True, d_phi=phi_t)

print(f'{'='*80}')
print('PAUL example 2.2.1')
tt = np.linspace(t_span[0], t_span[1], 100)
realsol = np.array([real_sol(t) for t in tt])
sol = [solver.eta(i) for i in tt]
print("max", np.max(np.abs(np.squeeze(sol) - np.squeeze(realsol))))
solution = np.array([real_sol(t) for t in solver.t])
print('solution', solution)
print('shape solver.y', solver.y)
print('adnaed', np.max(np.squeeze(solver.y) - np.squeeze(solution)))


print('sol', len(sol))
print('realsol', len(realsol))
plt.plot(tt, realsol, color="red", label='real solution')
plt.plot(tt, sol, color="blue", label='aproxx')
plt.legend()
plt.show()
