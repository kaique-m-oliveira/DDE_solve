import numpy as np
from DDE_solver.rkh_refactor import *

# WARN: STATE EXAMPLE


def f(t, y, x, z):
    f1 = x[0][0] + z[1][0] - z[0][1]
    f2 = 2*x[0][0] + x[1][1] + z[1][0]
    return [f1, f2]


def phi(t):
    return [np.cos(t), np.sin(t)]


def phi_t(t):
    return [-np.sin(t), np.cos(t)]


def alpha(t, y):
    return [t - np.pi/2, t - np.pi]


def real_sol(t):
    if 0 <= t <= np.pi/2:
        return [2 - np.cos(t), 2 - 2*np.cos(t)]
    if np.pi/2 <= t <= np.pi:
        return [2*t + 2*np.sin(t) - np.pi, 2*(2*t + np.cos(t) + 1 - np.pi)]


t_span = [0, np.pi]


solver = solve_dde(f, alpha, phi, t_span, beta=alpha,
                   neutral=True, d_phi=phi_t)

print(f'{'='*80}')
print('PAUL example 2.4.1')
tt = np.linspace(t_span[0], t_span[1], 100)
realsol = np.array([real_sol(t) for t in tt])
sol = [solver.eta(i) for i in tt]
print("max", np.max(np.abs(np.squeeze(sol) - np.squeeze(realsol))))
solution = np.array([real_sol(t) for t in solver.t])
print('adnaed', np.max(np.squeeze(solver.y) - np.squeeze(solution)))


print('sol', len(sol))
print('realsol', len(realsol))
plt.plot(tt, realsol, color="red", label='real solution')
plt.plot(tt, sol, color="blue", label='aproxx')
plt.legend()
plt.show()
