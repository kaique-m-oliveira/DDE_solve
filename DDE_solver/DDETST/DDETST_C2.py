import numpy as np

from DDE_solver.rkh_refactor import *


def f(t, y, x):
    y1, y2 = y
    x1, _ = x  # x = [y1(t - y2(t)), y2(t - y2(t))], but only x1 is used
    dy1 = -2 * x1
    dy2 = (abs(x1) - abs(y1)) / (1 + abs(x1))
    return [dy1, dy2]


def phi(t):
    return [1.0, 0.5]


def alpha(t, y):
    y1, y2 = y
    return t - y2


def real_sol(t):
    # No known analytical solution
    return [np.nan, np.nan]


t_span = [0, 40]


print(f'{'='*80}')
print(f''' {'='*80} 
      This is problem 1.2.6 from Paul
      ''')
methods = ['RKC3', 'RKC4', 'RKC5']
tolerances = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10]


for method in methods:
    for Tol in tolerances:
        solution = solve_dde(f, alpha, phi, t_span, method = method, Atol=Tol, Rtol=Tol)

        print('==========Counting============')
        print(f'method = {method}')
        print(f'Tol = {Tol}')
        print('steps: ', solution.steps)
        print('fails: ', solution.fails)
        print('feval: ', solution.feval)
        

t_plot = np.linspace(t_span[0], t_span[-1], 1000)
approx_plot =  [solution.eta(i) for i in t_plot]
plt.plot(t_plot, approx_plot, color="blue", label='aproxx')
plt.legend()
plt.show()
