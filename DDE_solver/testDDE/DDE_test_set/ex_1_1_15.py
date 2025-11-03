import numpy as np

from DDE_solver.rkh_refactor import *


def f(t, y, x):
    return -1000*y + 997*np.exp(-3)*x + (1000 - 997*np.exp(-3))

def phi(t):
    return 1 + np.exp(-3*t)

def alpha(t, y):
    return t - 1

# No analytical solution found 
def real_sol(t):
    return 1 + np.exp(-3*t)

t_span = [0, 5]


print(f'{'='*80}')
print(f''' {'='*80} 
      This is problem 1.1.15 from Paul
      ''')

methods = ['RKC3', 'RKC4', 'RKC5']
tolerances = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10]

for method in methods:
    for Tol in tolerances:
        solution = solve_dde(f, alpha, phi, t_span, method = method, Atol=Tol, Rtol=Tol)
        max = 0
        for i in range(len(solution.t) - 1):
            tt = np.linspace(solution.t[i], solution.t[i + 1], 100)
            sol = np.array([solution.eta(i) for i in tt])
            realsol = np.array([real_sol(i) for i in tt])
            max_diff = np.max(np.abs(realsol - sol))
            if max_diff > max:
                max = max_diff

        print('==========Counting============')
        print(f'method = {method}')
        print(f'Tol = {Tol}')
        print('number of steps: ', Counting.steps)
        print('number of fails: ', Counting.fails)
        print('number of fnc calls: ', Counting.fnc_calls)
        print('max diff', max)

t_plot = np.linspace(t_span[0], t_span[-1], 1000)
approx_plot =  [solution.eta(i) for i in t_plot]
realsol_plot = [real_sol(i) for i in t_plot]
plt.plot(t_plot, approx_plot, color="blue", label='aproxx')
plt.plot(t_plot, realsol_plot, color="red", label='real sol')
plt.legend()
plt.show()
