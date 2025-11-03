import numpy as np

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


t_span = [0, 5]


print(f'{'='*80}')
print(f''' {'='*80} 
      This is problem 1.1.10 from Paul
      ''')

methods = ['RKC3', 'RKC4', 'RKC5']
tolerances = [1e-2, 1e-3,  1e-4, 1e-6, 1e-8, 1e-10]
real_discs = np.array([0, 1, 2, 3, 4])


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
        
        disc_diff = np.max(np.abs(real_discs[:len(solution.discs)] - np.array(solution.discs)))

        print('==========Counting============')
        print(f'method = {method}')
        print(f'Tol = {Tol}')
        print('number of steps: ', Counting.steps)
        print('number of fails: ', Counting.fails)
        print('number of fnc calls: ', Counting.fnc_calls)
        print('max diff', max_diff)
        print('number_of_discs_found', len(solution.discs))
        print('disc diff', disc_diff)
        print('discs', solution.discs)

t_plot = np.linspace(t_span[0], t_span[-1], 1000)
approx_plot =  [solution.eta(i) for i in t_plot]
realsol_plot = [real_sol(i) for i in t_plot]
plt.plot(t_plot, approx_plot, color="blue", label='aproxx')
plt.plot(t_plot, realsol_plot, color="red", label='real sol')
plt.legend()
plt.show()
