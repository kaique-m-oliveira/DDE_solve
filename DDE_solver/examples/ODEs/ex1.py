from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np


def f(t, y):
    return -y


t_span = [1, 2]
y0 = [1]  # Change y0 to a 1D NumPy array
sol = solve_ivp(f, t_span)

# Use .T to ensure proper shape for plotting
plt.plot(sol.t, sol.y.T, color="red")
plt.show()
