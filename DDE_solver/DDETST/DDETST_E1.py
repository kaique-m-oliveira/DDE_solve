import numpy as np

from DDE_solver.rkh_refactor import *

r, c = np.pi/np.sqrt(3) + 1/20, np.sqrt(3)/(2*np.pi) - 1/25
def f(t, y, x, z):
    return r*y*(1 - x - c*z)

def phi(t):
    return 2 + t

def phi_t(t):
    return 1

def alpha(t, y):
    return t - 1

beta = alpha

def real_sol(t):
    return np.nan

t_span = [0, 40]

print(f'{'='*80}')
print(f''' {'='*80} 
      This is problem DDETST H2 
      ''')

methods = ['RKC3', 'RKC4', 'RKC5']
tolerances = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10]


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
