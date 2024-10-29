import ast
import inspect
import random

import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicHermiteSpline, CubicSpline
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
            return sol[i](x)
    print(f"{x} is not in the interval t")


def f(t, y, yq):
    return -yq


def delay(t):
    return t - np.pi / 2


def phi(t):
    return np.sin(t)


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
    k1 = f(t0, y0, yq)
    k2 = f(t0 + h / 2, y0 + (h * k1) / 2, yq + (h * k1) / 2)
    k3 = f(t0 + h / 2, y0 + (h * k2) / 2, yq + (h * k2) / 2)
    k4 = f(t0 + h, y0 + (h * k3), yq + (h * k3))
    y1 = y0 + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    # print(f"y = {y1} , t0 {t0}, t1 {t1}, y0 {y0} , yq {yq}")
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


def rk4_cont(f, t_span, phi, delay, h):
    """who tf knows"""
    discs = get_primary_discontinuities(t_span, [delay])[0]
    sol = [phi]

    t0 = t_span[0]
    y = [phi(t0)]
    t = [t0]
    dy = [0]
    # print(f"y = {len(y)} dy = {len(dy)}")
    for disc in discs:
        y = [y[-1]]
        t = [t[-1]]
        dy = [dy[-1]]
        while delay(t[-1]) < disc:  # adding to the discrete solution
            x = delay(t[-1])
            # print(f"x is {x} and t is {t[-1]}")
            yq = get_yq(discs, x, sol)
            y.append(rk4_arit_delay(f, t[-1], t[-1] + h, y[-1], yq))
            dev = (y[-1] - y[-2]) / h
            print(f"t = {t[-1]} and dy = {dev}")
            dy.append(dev)
            t.append(t[-1] + h)
            # print(f" dy at t = {t[-1]} is {(sol[-1](t0 + h) - sol[-1](t0)) / h}")
            # print(f" len(y) {len(y)} len(dy) {len(dy)}")
        # print(f"y = {len(y)} dy = {len(dy)}")
        # sol.append(CubicSpline(t, y))
        sol.append(CubicHermiteSpline(t, y, dy))
        # print(f"começo e final de t, {t[0]} e {t[-1]}")

    def solution(var):
        for i in range(len(discs)):
            if var <= discs[i]:
                return sol[i](var)

    return solution


t_span = [0, 10]
y0 = 2
# t = np.arange(0, 100, 0.1)
h = 0.01
sol = rk4_cont(f, t_span, phi, delay, h)
t = np.arange(-1, 10, 0.1)


y = [sol(i) for i in t]
sin = np.sin(t)
# WARN: this error analysis suggests the method is linear
error = 0
for i in range(100):
    x = random.uniform(0, 10)
    diff = abs(np.sin(x) - sol(x))
    # print(diff)
    if diff > error:
        error = diff

# print("len of t", len(t))
print("this is the max error", error)
plt.plot(t, y)
plt.plot(t, sin)
plt.show()


# # WARN:  THIS IS NOT GONNA WORK WITH SOLVE_IVP UNLESS I FIGURE OUT HOW TO USE THE PHI INSIDE F
# def DDE_solve(fun, t_span, phi, t_eval):
#     delays = find_delay(fun)
#     discs = get_primary_discontinuities(t_span, delays)[0]
#     sol = [phi]
#     N = len(discs)
#     for i in range(N):
#         sol = solve_ivp(fun, t_span, sol[i](discs[i]), method="RK45", t_eval=t_eval)
#     return sol


# def get_disc_euler(t_span, delay):
#     t0, tf = t_span
#     h = 10
#     t = t0
#     f = lambda x: delay(x) - t
#     discs = [t0]
#     while t < tf and h > 10**-5:
#         disc = fsolve(f, t + 0.01)[0]
#         if disc < t:
#             print("disc < t, deu ruim")
#             return
#         discs.append(disc)
#         h = disc - t
#         t = disc
#     return discs


def g(t, y):
    return 3 * y


# if __name__ == "__main__":
# phi = lambda t: 1
# t = np.arange(0, 100, 0.01)
# delay = find_delay(f)
# print(delay)
# discs = get_primary_discontinuities([0, 100], delay)[0]
# print(discs)
