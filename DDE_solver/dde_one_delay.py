import ast
import inspect

import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
from scipy.optimize import fsolve


def find_delay(f):
    """returns a list with the delays"""
    source_code = inspect.getsource(f)
    tree = ast.parse(source_code)

    delay_funcs, delay_strings = [], []

    for node in ast.walk(tree):
        if isinstance(node, ast.Return):
            # Find all Call nodes within the return statement
            call_nodes = [n for n in ast.walk(node.value) if isinstance(n, ast.Call)]
            for call_node in call_nodes:
                arg_node = call_node.args[0]
                if isinstance(arg_node, ast.BinOp):
                    delay_string = ast.unparse(arg_node)
                    delay_funcs.append(lambda t, f=delay_string: eval(f))
                    delay_strings.append(delay_string)
    print("the delays are:", delay_strings)
    return delay_funcs


def get_primary_discontinuities(t_span, delays):
    """returns the primary discontinuities in the interval t from the delays"""
    t0, tf = t_span
    N = len(delays)
    discontinuities = [[t0] for i in range(N)]
    for i in range(N):
        h = 10
        t = t0
        f = lambda x: delays[i](x) - t
        while t < tf and h > 10**-5:
            disc = fsolve(f, t + 0.01)[0]
            if disc < t:
                print("disc < t, deu ruim")
                return

            discontinuities[i].append(disc)
            h = disc - t
            t = disc
            # print(disc)
            # print("t", t, "tf", tf, "t < tf", t < tf)
            # print(f"h = {h} is less then 10**-5 {h < 10**-5}")
    return discontinuities


def get_yq(discs, x, sol):

    for i in range(len(discs)):
        if x <= discs[i]:
            # print("-" * 20)
            # print(f"x = {x}, i = {i}")
            # print(f"discs {discs}")
            # print(f"sol {sol}")
            return sol[i](x)
    print(f"{x} is not in the interval t")


# WARN:  THIS IS NOT GONNA WORK WITH SOLVE_IVP UNLESS I FIGURE OUT HOW TO USE THE PHI INSIDE F
def DDE_solve(fun, t_span, phi, t_eval):
    delays = find_delay(fun)
    discs = get_primary_discontinuities(t_span, delays)[0]
    sol = [phi]
    N = len(discs)
    for i in range(N):
        sol = solve_ivp(fun, t_span, sol[i](discs[i]), method="RK45", t_eval=t_eval)
    return sol


def get_disc_euler(t_span, delay):
    t0, tf = t_span
    h = 10
    t = t0
    f = lambda x: delay(x) - t
    discs = [t0]
    while t < tf and h > 10**-5:
        disc = fsolve(f, t + 0.01)[0]
        if disc < t:
            print("disc < t, deu ruim")
            return
        discs.append(disc)
        h = disc - t
        t = disc
    return discs


def f(t, y, yq):
    return yq


def delay(t):
    return t - 1


def phi(t):
    return 1


def rk4_arit(f, t0, t1, y0):
    h = t1 - t0
    k1 = h * f(t0, y0)
    k2 = h * f(t0 + h / 2, y0 + h * k1 / 2)
    k3 = h * f(t0 + h / 2, y0 + h * k2 / 2)
    k4 = h * f(t0 + h, y0 + h * k3)
    y1 = y0 + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return y1


def rk4_arit_delay(f, t0, t1, y0, yq):
    # print("rk4_arit_delay: ", "yq", yq, "f(t0, y0, yq)", f(t0, y0, yq))

    h = t1 - t0
    k1 = h * f(t0, y0, yq)
    k2 = h * f(t0 + h / 2, y0 + h * k1 / 2, yq + h * k1 / 2)
    k3 = h * f(t0 + h / 2, y0 + h * k2 / 2, yq + h * k2 / 2)
    k4 = h * f(t0 + h, y0 + h * k3, yq + h * k3)
    y1 = y0 + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return y1


def rk4(f, t_span, y0, t):
    """as the name suggests"""

    N = len(t)
    y = np.zeros(N)
    y[0] = y0
    for n in range(N - 1):
        y[n + 1] = rk4_arit(f, t[n], t[n + 1], y[n])
    # print("len(t)", len(t), "len(t)", len(y))
    # print(print(t), print(t))
    return y


def find_yq(t, t0, yq):
    return


def rk4_cont(f, t_span, phi, delay, h):
    """who tf knows"""
    discs = get_primary_discontinuities(t_span, [delay])[0]
    sol = [phi]

    y = [phi(t_span[0])]
    t = [t_span[0]]
    for disc in discs:
        y = [y[-1]]
        t = [t[-1]]
        while delay(t[-1]) < disc:  # adding to the discrete solution
            yq = get_yq(discs, delay(t[-1]), sol)
            y.append(rk4_arit_delay(f, t[-1], t[-1] + h, y[-1], yq))
            t.append(t[-1] + h)
        sol.append(CubicSpline(t, y))
        yyy = [sol[-1](i) for i in t]
        plt.plot(t, yyy)
        plt.show()

    def solution(var):
        for i in range(len(discs)):
            if var <= discs[i]:
                return sol[i](var)

    return solution


t_span = [0, 5]
y0 = 2
# t = np.arange(0, 100, 0.1)
h = 0.01
sol = rk4_cont(f, t_span, phi, delay, h)
t = np.arange(-1, 5, 0.1)

y = [sol(i) for i in t]
plt.plot(t, y)
plt.show()


# def test(t):
#     def testing(t):
#         return 2 * t
#
#     return testing
#
#
# bla = test(2)
# print("bla", bla(8))


def g(t, y):
    return 3 * y


# if __name__ == "__main__":
# phi = lambda t: 1
# t = np.arange(0, 100, 0.01)
# delay = find_delay(f)
# print(delay)
# discs = get_primary_discontinuities([0, 100], delay)[0]
# print(discs)
