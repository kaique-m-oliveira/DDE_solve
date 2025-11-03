import numpy as np

from DDE_solver.rkh_refactor import *

# Parameters
r1 = 0.02
r2 = 0.005
alpha_param = 3.0
delta = 0.01


def f(t, y, x):
    """
    Standard convention for a single delay:
    - x is the full delayed state vector evaluated at alpha(t,y) = t - y4(t)
    """
    y1, y2, y3, y4 = y
    xd1, xd2, xd3, xd4 = x  # delayed state: y(t - y4(t))

    dy1 = -r1 * y1 * y2 + r2 * y3
    dy2 = -r1 * y1 * y2 + alpha_param * r1 * xd1 * xd2
    dy3 = r1 * y1 * y2 - r2 * y3

    denom = xd1 * xd2 + xd3
    # Protect against division by zero (solver should handle, but keep safe)
    if denom == 0.0:
        dy4 = 1.0 + 0.0  # fallback; you may prefer np.inf or raise
    else:
        dy4 = 1.0 + ((3.0 * delta - y1 * y2 - y3) / denom) * np.exp(delta * y4)

    return [dy1, dy2, dy3, dy4]


def phi(t):
    # history for t <= 0
    return [5.0, 0.1, 0.0, 0.0]


def alpha(t, y):
    # single state-dependent delay: t - y4(t)
    y1, y2, y3, y4 = y
    return t - y4


def real_sol(t):
    # no closed-form solution provided
    return [np.nan, np.nan, np.nan, np.nan]


t_span = [0.0, 40.0]


print(f'{'='*80}')
print(f''' {'='*80} 
      This is problem 1.1.10 from Paul
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
