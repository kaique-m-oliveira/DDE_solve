import numpy as np

from DDE_solver.rkh_refactor import *


def f(t, y, x, z):
    return np.cos(t)*(1 + x) + 0.6 * y* z


def phi(t):
    return -t/2


def phi_t(t):
    return -1/2


def alpha(t, y):
    return t*y**2

beta = alpha


epsilon = 0
t_span = [0.25, 4.1]


print(f'{'='*80}')
print(f''' {'='*80} 
      This is problem 1.3.4 from Paul
      ''')

methods = ['RKC3', 'RKC4', 'RKC5']
tolerances = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10]


for method in methods:
    for Tol in tolerances:
        print('method', method)
        # solution = solve_ndde(t_span, f, alpha, beta, phi, phi_t, method = method, Atol=Tol, Rtol=Tol)
        solution = solve_dde(f, alpha, phi, t_span, neutral = True, d_phi=phi_t, beta=beta, method = method, Atol=Tol, Rtol=Tol)
        
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
