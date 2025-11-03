import numpy as np

from DDE_solver.rkh_refactor import *


def f(t, y, yq):
    return np.exp(yq)/t


def phi(t):
    return 0


def alpha(t, y):
    return y - np.log(2) + 1


def real_sol(t):
    if 1 <= t <= 2:
        return np.log(t)
    elif 2 <= t <= 4:
        return t/2 + np.log(2) - 1


print(f'{'='*80}')
print(f''' {'='*80} 
      This is problem 1.3.1 from Paul
      ''')

t_span = [1, 4]
methods = ['RKC3', 'RKC4', 'RKC5']
tolerances = [1e-2,  1e-4, 1e-6, 1e-8, 1e-10]


for method in methods:
    for Tol in tolerances:
        solution = solve_dde(f, alpha, phi, t_span, method = method, Atol=Tol, Rtol=Tol)
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
        print('steps: ', solution.steps)
        print('fails: ', solution.fails)
        print('feval: ', solution.feval)
        print('max diff', max_diff)

t_plot = np.linspace(t_span[0], t_span[-1], 1000)
approx_plot =  [solution.eta(i) for i in t_plot]
realsol_plot = [real_sol(i) for i in t_plot]
plt.plot(t_plot, approx_plot, color="blue", label='aproxx')
plt.plot(t_plot, realsol_plot, color="red", label='real sol')
plt.legend()
plt.show()
