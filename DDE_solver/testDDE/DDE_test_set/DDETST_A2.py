import numpy as np

from DDE_solver.rkh_refactor import *


def f(t, y, x):
    y1, y2 = y
    x1, x2 = x
    dy1 = 1.1/(1 + np.sqrt(10)*x1**(5/4)) - 10*y1/(1 + 40*y2)
    dy2 =100*y1/(1+40*y2) - 2.43*y2
    return dy1, dy2

def phi(t):
    return [1.05767027/3, 1.030713491/3]

def alpha(t, y):
    return t - 20

# No analytical solution found 


t_span = [0, 100]


solver = solve_dde(f, alpha, phi, t_span, Atol=1e-8, Rtol=1e-8)

print(f'{'='*80}')
print(f''' {'='*80} 
      This is problem A2
      ''')
tt = np.linspace(t_span[0], t_span[1], 100)
sol = [solver.eta(i) for i in tt]

print('==========Counting============')
print('number of steps: ', Counting.steps)
print('number of fails: ', Counting.fails)
print('number of fnc calls: ', Counting.fnc_calls)

plt.plot(tt, sol, color="blue", label='aproxx')
plt.legend()
plt.show()
