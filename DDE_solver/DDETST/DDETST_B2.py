import numpy as np

from DDE_solver.rkh_refactor import *


def f(t, y, x):
    fx = 1.0 if x < 0 else -1.0
    return fx - y


def phi(t):
    return 1.0


def alpha(t, y):
    return t / 2.0


def real_sol(t):
    if 0 <= t <= 2*np.log(2):
        return 2*np.exp(-t) - 1
    elif 2*np.log(2) < t <= 2*np.log(6):
        return 1 - 6*np.exp(-t)
    elif 2*np.log(6) < t <= 2*np.log(66):
        return 66*np.exp(-t) - 1
    else:
        return np.nan  # outside domain


t_span = [0, 2*np.log(66)]

print('alpha0', alpha(0.5, phi(0.5)))


print(f'{'='*80}')
print(f''' {'='*80} 
      This is problem 1.2.6 from Paul
      ''')
methods = ['RKC3', 'RKC4', 'RKC5']
tolerances = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10]


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

            yyy = np.array([real_sol(i) for i in solution.t])
        
        diff_points = np.max(np.abs(yyy - solution.y))
        print('==========Counting============')
        print(f'method = {method}')
        print(f'Tol = {Tol}')
        print('steps: ', solution.steps)
        print('fails: ', solution.fails)
        print('feval: ', solution.feval)
        print('max diff', max_diff)
        print('diff malha', diff_points)
        diff_points = 0
        

t_plot = np.linspace(t_span[0], t_span[-1], 1000)
approx_plot =  [solution.eta(i) for i in t_plot]
plt.plot(t_plot, approx_plot, color="blue", label='aproxx')
plt.legend()
plt.show()
