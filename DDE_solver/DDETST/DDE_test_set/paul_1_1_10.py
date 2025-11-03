import numpy as np

from DDE_solver.rkh_refactor import *


def f(t, y, x):
    return x*y

def phi(t):
    if t < -np.pi/2:
        return 0
    elif -np.pi/2 <= t < 0:
        return -2
    else:
        return -1

def alpha(t, y):
    return t - np.pi

def real_sol(t):
    if 0 <= t <= np.pi/2:
        return -1
    elif np.pi/2 <= t <= np.pi:
        return -np.exp(np.pi - 2*t)
    elif np.pi <= t <= 3*np.pi/2:
        return  -np.exp(-t)
    elif 3*np.pi/2 <= t <= 6:
        return -np.exp(-3*np.pi/2 + (np.exp(3*np.pi - 2*t) - 1)/2)

t_span = [0, 6]


print(f'{'='*80}')
print(f''' {'='*80} 
      This is problem 1.1.10 from Paul
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
        print('discs', solution.discs)
        input('here')

t_plot = np.linspace(t_span[0], t_span[-1], 1000)
approx_plot =  [solution.eta(i) for i in t_plot]
realsol_plot = [real_sol(i) for i in t_plot]
plt.plot(t_plot, approx_plot, color="blue", label='aproxx')
plt.plot(t_plot, realsol_plot, color="red", label='real sol')
plt.legend()
plt.show()
