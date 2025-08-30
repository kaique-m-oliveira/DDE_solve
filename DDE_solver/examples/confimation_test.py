import numpy as np
# WARN: this one is mine


def f(t, y, x1, x2, x3, x4):
    return -x1 + x2 - x3*x4


def real_sol(t):
    if t < 0:
        return 1
    elif 0 <= t <= 1:
        return -t
    elif 1 < t <= 2:
        return (1/2) * t**2 - t - (1/2)
    elif 2 < t <= 3:
        return (-1/6) * t**3 + (1/2) * t**2 - (7/6)
    elif 3 < t <= 4:
        return (1/24) * t**4 - (1/6) * t**3 - (1/4) * t**2 + t - (19/24)
    elif 4 < t <= 5:
        return (-1/120) * t**5 + (1/6) * t**4 - (5/3) * t**3 + (109/12) * t**2 - 24 * t + (2689/120)
    else:
        return np.nan


def alpha(t, y):
    return [t-1,  t-2, t-3, t-4]


def alpha1(t, y):
    return t-1


def alpha2(t, y):
    return t-2


def alpha3(t, y):
    return t-3


def alpha4(t, y):
    return t-4


tn = 0.95
h = 0.05
yn = real_sol(tn)


def rk4_step(f, tn, yn, h, alphas, real_sol):
    alpha1, alpha2, alpha3, alpha4 = alphas

    t1 = tn
    y1 = yn
    Y11 = real_sol(alpha1(t1, y1))
    Y12 = real_sol(alpha2(t1, y1))
    Y13 = real_sol(alpha3(t1, y1))
    Y14 = real_sol(alpha3(t1, y1))
    k1 = f(t1, y1, Y11, Y12, Y13, Y14)

    t2 = tn + 0.5 * h
    y2 = yn + 0.5 * h * k1
    Y21 = real_sol(alpha1(t2, y2))
    Y22 = real_sol(alpha2(t2, y2))
    Y23 = real_sol(alpha3(t2, y2))
    Y24 = real_sol(alpha3(t2, y2))
    k2 = f(t2, y2, Y21, Y22, Y23, Y24)

    t3 = tn + 0.5 * h
    y3 = yn + 0.5 * h * k2
    Y31 = real_sol(alpha1(t3, y3))
    Y32 = real_sol(alpha2(t3, y3))
    Y33 = real_sol(alpha3(t3, y3))
    Y34 = real_sol(alpha3(t3, y3))
    k3 = f(t3, y3, Y31, Y32, Y33, Y34)

    t4 = tn + h
    y4 = yn + h * k2
    Y41 = 1
    # Y41 = real_sol(alpha1(t4, y4))
    Y42 = real_sol(alpha2(t4, y4))
    Y43 = real_sol(alpha3(t4, y4))
    Y44 = real_sol(alpha3(t4, y4))
    k4 = f(t4, y4, Y41, Y42, Y43, Y44)

    y_next = yn + h * (k1/6 + k2/3 + k3/3 + k4/6)

    return y_next


alphas = [alpha1, alpha2, alpha3, alpha4]
y1 = rk4_step(f, tn, yn, h, alphas, real_sol)
y1_exact = real_sol(tn + h)

print('y1 =', y1)
print('y(t1)', y1_exact)
print('ERROR at t = {tn + h}', abs(y1 - y1_exact))
