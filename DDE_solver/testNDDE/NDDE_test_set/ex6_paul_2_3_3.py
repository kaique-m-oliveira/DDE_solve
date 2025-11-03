import numpy as np

from DDE_solver.rkh_refactor import *


def f(t, y, x, z):
    return np.cos(t)*(1 + x) + y*z - np.sin(t + t*np.sin(t)**2)


def phi(t):
    return 0


def phi_t(t):
    return 1


def alpha(t, y):
    return t*y**2


def real_sol(t):
    if 0 <= t <= np.pi/2:
        return np.sin(t)


t_span = [0, np.pi/2]


print(f'{'='*80}')
print(f''' {'='*80} 
      This is problem 1.3.4 from Paul
      ''')

methods = ['RKC3', 'RKC4', 'RKC5']
tolerances = [1e-6, 1e-8, 1e-10]


for method in methods:
    for Tol in tolerances:
        print('method', method)
        # solution = solve_ndde(t_span, f, alpha, beta, phi, phi_t, method = method, Atol=Tol, Rtol=Tol)
        solution = solve_dde(f, alpha, phi, t_span, neutral = True, d_phi=phi_t, beta=beta, method = method, Atol=Tol, Rtol=Tol)
        max_diff = 0
        for i in range(len(solution.t) - 1):
            tt = np.linspace(solution.t[i], solution.t[i + 1], 100)
            sol = np.array([solution.eta(i) for i in tt])
            realsol = np.array([real_sol(i) for i in tt])
            max_diff = np.max(np.abs(realsol - sol))
            if max_diff > max_diff:
                max_diff = max_diff
        
        print('==========Counting============')
        print(f'method = {method}')
        print(f'Tol = {Tol}')
        print('number of steps: ', Counting.steps)
        print('number of fails: ', Counting.fails)
        print('number of fnc calls: ', Counting.fnc_calls)
        print('max diff', max_diff)

t_plot = np.linspace(t_span[0], t_span[-1], 1000)
approx_plot =  [solution.eta(i) for i in t_plot]
realsol_plot = [real_sol(i) for i in t_plot]
plt.plot(t_plot, approx_plot, color="blue", label='aproxx')
plt.plot(t_plot, realsol_plot, color="red", label='real sol')
plt.legend()
plt.show()
