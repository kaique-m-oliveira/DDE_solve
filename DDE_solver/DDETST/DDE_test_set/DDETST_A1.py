import numpy as np

from DDE_solver.rkh_refactor import *


def f(t, y, x):
    return 0.2*x/(1+x**10) - 0.1*y

def phi(t):
    return 0.5

def alpha(t, y):
    return t - 14

# No analytical solution found 


t_span = [0, 500]


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
