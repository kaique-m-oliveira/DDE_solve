import numpy as np
from DDE_solver.rkh_refactor import *


def f(t, y, yq):
    yq1, yq2 = yq
    return yq1 + yq2 + 1/2


def phi(t):
    return 1 if t <= -1 else 0


def alpha(t, y):
    return [t - abs(y) - 1, t - 2*y - 2]


t_span = [0, 2]
# discs = [(-1, 1, 0)]

solutionList = solve_dde(f, alpha, phi, t_span)  # , discs=discs)


print(f'{'='*80}')
tt = np.linspace(t_span[0], t_span[1], 100)
# sol1 = [solutionList.solutions[0].eta(i) for i in tt]
# sol2 = [solutionList.solutions[1].eta(i) for i in tt]
# plt.plot(tt, sol1, color="blue", label='aproxx1')
# plt.plot(tt, sol2, color="black", label='aproxx2')
plt.plot(solutionList.t, solutionList.y, label="fake")
plt.legend()
plt.show()
