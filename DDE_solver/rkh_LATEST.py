import time
import numbers
from bisect import bisect_right
from dataclasses import dataclass, field
import random
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import lu_factor, lu_solve, norm
from scipy.optimize import root
from scipy.integrate import solve_ivp


@dataclass
class CRKParameters:
    theta1: float = 1 / 3
    TOL: float = 1e-5
    rho: float = 0.9
    omega_min: float = 0.5
    omega_max: float = 1.5
    A: np.ndarray = field(default_factory=lambda: np.array([
        [0,   0,   0, 0],
        [1/2, 0,   0, 0],
        [0,  1/2,  0, 0],
        [0,   0,   1, 0]
    ], dtype=float))
    b: np.ndarray = field(default_factory=lambda: np.array(
        [1/6, 1/3, 1/3, 1/6], dtype=float))
    c: np.ndarray = field(default_factory=lambda: np.array(
        [0, 1/2, 1/2, 1], dtype=float))


def vectorize_func(func):
    def wrapper(*args, **kwargs):
        # return np.array(func(*args, **kwargs))
        return np.atleast_1d(func(*args, **kwargs))
    return wrapper


def real_alpha(t, y):
    return t - 1


def real_sol(t):
    result = []
    if isinstance(t, (float, int)):
        t = [t]
    for ti in t:
        if ti < 0:
            result.append(1)
        elif 0 <= ti <= 1:
            result.append(-ti)
        elif 1 < ti <= 2:
            result.append((1/2) * ti**2 - ti - (1/2))
        elif 2 < ti <= 3:
            result.append((-1/6) * ti**3 + (1/2) * ti**2 - (7/6))
        elif 3 < ti <= 4:
            result.append((1/24) * ti**4 - (1/6) * ti **
                          3 - (1/4) * ti**2 + ti - (19/24))
        elif 4 < ti <= 5:
            result.append((-1/120) * ti**5 + (1/6) * ti**4 - (5/3)
                          * ti**3 + (109/12) * ti**2 - 24 * ti + (2689/120))
        else:
            result.append(np.nan)
    return np.array(result)


def validade_arguments(f, alpha, phi, t_span, d_f, d_alpha, d_phi):
    t0, tf = map(float, t_span)
    t_span = [t0, tf]
    y0 = phi(t0)

    if isinstance(y0, numbers.Real) or np.isscalar(y0):
        ndim = 1
    elif isinstance(y0, (list, np.ndarray)):
        ndim = len(y0)
    else:
        raise TypeError(f"Unsupported type for phi(t0): {type(y0)}")

    alpha0 = alpha(t0, y0)
    if isinstance(alpha0, numbers.Real) or np.isscalar(alpha0):
        ndelays = 1
    elif isinstance(alpha0, (list, np.ndarray)):
        ndelays = len(alpha0)
    else:
        raise TypeError(f"Unsupported type for alpha(t0, phi(t0)): {alpha0}")

    f = vectorize_func(f)
    alpha = vectorize_func(alpha)
    phi = vectorize_func(phi)

    if (d_f is not None) and (d_alpha is not None) and (d_phi is not None):

        df = [vectorize_func(func) for func in d_f]
        d_alpha = [vectorize_func(func) for func in d_alpha]
        d_phi = vectorize_func(d_phi)
        return ndim, ndelays, f, alpha, phi, t_span, d_f, d_alpha, d_phi

    d_f = [None, None, None]
    d_alpha = [None, None]
    d_phi = [None, None]
    return ndim, ndelays, f, alpha, phi, t_span, d_f, d_alpha, d_phi


class OneStep:

    def __init__(self, problem, solution, h, n_stages=8):
        tn = solution.t[-1]
        yn = solution.y[-1]
        self.problem = problem
        self.solution = solution
        self.h = h
        self.t = [tn, tn + self.h]
        self.h_next = None
        self.y = [yn, 1]  # we don't have yn_plus yet
        self.n = problem.ndim
        self.y_tilde = None
        self.K = np.zeros((n_stages, self.n), dtype=float)
        self.eta = np.zeros(2)
        self.eta_t = np.zeros(2)
        self.disc_local_error = None
        self.uni_local_error = None
        self.params = CRKParameters()
        self.overlap = False
        self.test = False
        self.disc = False
        self.ndim = problem.ndim
        self.ndelays = problem.ndelays

    @property
    def eeta(self):
        def eval(t):
            t = np.atleast_1d(t)  # accept scalar or array

            results = []
            for ti in t:
                if ti <= self.t[0]:
                    results.append(self.solution.eta(ti))
                else:
                    results.append(self._hat_eta_0(ti))
            return np.squeeze(results)
        return eval

    def is_there_disc(self):
        tn, h, yn = self.t[0], self.h, self.y[0]
        f, eta, alpha = self.problem.f, self.solution.etas[-1], self.problem.alpha
        discs = self.solution.discs
        hn = self.solution.t[-1] - self.solution.t[-2]
        # FIX: maybe not work as well
        if hn <= 1e-15:
            return False
        theta = 1 + h/hn

        def d_zeta(t, disc):
            return alpha(t, eta(t)) - np.full(self.ndelays, disc)

        disc = None
        for disc in discs:
            is_delay = d_zeta(tn, disc) * d_zeta(tn + theta * h, disc) < 0
            # print('is delay', is_delay)
            disc_position = d_zeta(tn, disc) * d_zeta(tn + theta * h, disc) < 0
            if np.any(disc_position):
                new_disc = self.get_disc(disc, disc_position)
                # print(
                #     '//////////////////////////////////////////////////////////////////////////')
                # print('DISC POSITION', disc_posi)
                # print('NEW DISCS', new_discs)
                # new_disc = new_discs[disc_position]
                # print('NEW DISC', new_disc)
                self.disc = new_disc
                return True
        return False

    def get_disc(self, disc, disc_position):
        rho, TOL = self.params.rho, self.params.TOL
        alpha = self.problem.alpha
        alpha_t, alpha_y = self.problem.d_alpha
        eta, eta_t = self.solution.etas[-1], self.solution.etas_t[-1]
        iter, max_iter = 0, 30
        # t = np.full(self.problem.ndelays, self.t[0] + self.h/2)
        # print('shape t', t.shape)
        t = self.t[0] + self.h/2

        # def d_zeta(t, disc):
        t_values = []
        for index in np.where(disc_position)[0].tolist():
            def d_zeta(t):
                # print('eta(t)', eta(t))
                a = alpha(t, eta(t))[index]
                b = disc
                return a - b

            # FIX: for now, let's use this
            sol = root(d_zeta, t, method='hybr')

            # def d_zeta_t(t, disc):
            #     print(f't {t}')
            #     a = alpha_t(t, eta(t))[index]
            #     print(f'a {a} shape {a.shape}')
            #     b = alpha_y(t, eta(t))[index]
            #     print(f'b {b} shape {b.shape}')
            #     c = eta_t(t)
            #     print(f'c {c} shape {c.shape}')
            #     return a + b * c
            # for i in range(max_iter):
            #     val = abs(d_zeta(t, disc))
            #     if np.any(val < rho*TOL):
            #         break
            #     a = -d_zeta(t, disc)
            # t += -d_zeta(t, disc)/ d_zeta_t(t, disc)

            t_values.append(sol.x)
        return min(t_values)

    def one_step_RK4(self):
        tn, h, yn = self.t[0], self.h, self.y[0]
        self.h = h
        self.t = [tn, tn + self.h]
        self.h_next = None
        self.y = [yn, 1]  # we don't have yn_plus yet
        self.n = self.problem.ndim
        self.y_tilde = None
        self.K = np.zeros((8, self.n), dtype=float)
        self.eta = np.zeros(2)
        self.eta_t = np.zeros(2)
        self.disc_local_error = None
        self.uni_local_error = None
        self.params = CRKParameters()
        self.overlap = False
        self.test = False
        self.disc = False

        # print('============================RK STEP================================')
        # print('tn = ', tn, 'h = ', h, 'tn + h = ', tn + h)
        # print("self.problem", self.problem)
        # print("self.solution", self.solution)
        # print("self.h", self.h)
        # print("self.t", self.t)
        # print("self.h_next", self.h_next)
        # print("self.y", self.y)
        # print("self.n", self.n)
        # print("self.y_tilde", self.y_tilde)
        # print("self.K", self.K)
        # print("self.eta", self.eta)
        # print("self.eta_t", self.eta_t)
        # print("self.disc_local_error", self.disc_local_error)
        # print("self.uni_local_error", self.uni_local_error)
        # print("self.params", self.params)
        # print("self.overlap", self.overlap)
        # print("self.test", self.test)
        # print("self.disc", self.disc)
        # print("self.ndim", self.ndim)
        # print("self.ndelays", self.ndelays)
        # f, eta, alpha = self.problem.f, self.solution.eta, self.problem.alpha
        f, eta, alpha = self.problem.f, self.solution.eta, self.problem.alpha
        c = self.params.c
        # print(f' TESTING eta(1) {eta(1)} realsol(1) {real_sol(1)}')
        realf = np.zeros(4)
        self.K[0] = f(tn, yn, eta(alpha(tn, yn)))
        # realf[0] = f(tn, yn, real_sol(alpha(tn, yn)))
        # print('ERROR K0', self.K[0] - realf[0])

        # WARN: this one is mine
        for i in range(1, 4):
            ti = tn + c[i] * h
            yi = yn + c[i] * h * self.K[i - 1]
            if np.all(alpha(ti, yi) <= np.full(self.ndelays, tn)):
                def real_alpha(t, y): return [t-1,  t-2, t-3, t-4]
                # print('ERROR alpha', max(alpha_i - real_alpha_i))
                alpha_i = alpha(ti, yi)
                real_alpha_i = alpha(ti, yi)
                Y_tilde = eta(alpha_i)
                # print('alpha_i', alpha_i)
                # print('Y_tilde', Y_tilde)
                # Y_ex = real_sol(alpha(ti, real_sol(ti)))
                # print(f'ERROR Y_tilde[{i}], {abs(Y_tilde - Y_ex)}')
                # print(f'SHAPES: Y_tilde {Y_tilde.shape} yn {yn.shape} c[i] {c[i].shape} yi {
                #       yi.shape} self.K[i - 1] {self.K[i-1].shape}')
                # Y_tilde = eta(alpha(ti, yi * self.K[i - 1]))
                self.K[i] = f(ti, yi, Y_tilde)
                # K_ex = f(ti, real_sol(ti), Y_ex)
                # print(f'ERROR K[{i}] {self.K[i] - K_ex}')
            else:  # this would be the overlapping case
                self.overlap = True
                success = self._simplified_Newton()
                print('successfull newton')
                if not success:
                    return False
                break
        # WARN: above is mine
            # self.K[i] = f(tn + c[i] * h, yn + c[i] * h * self.K[i-1], Y_tilde)
            # realf[i] = f(tn + c[i] * h, real_sol(tn + c[i] * h), Y_ex)

        self.y[1] = yn + h * (self.K[0] / 6 + self.K[1] /
                              3 + self.K[2] / 3 + self.K[3] / 6)
        # print('>>>>>>>>>>>>>>>>>>>>>>>check values<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        # print('y[0]', self.y[0], 'real_sol(tn)', real_sol(
        #     tn),  'ERROR:', self.y[0] - real_sol(tn))
        # print('y[1]', self.y[1], 'realsol(tn + h)', real_sol(tn + h),
        #       'ERROR:', self.y[1] - real_sol(tn + h))

        return True

    def _simplified_Newton(self):
        time1 = time.time()
        print('Newton', self.t)
        A, b, c = self.params.A, self.params.b, self.params.c
        rho, TOL = self.params.rho, self.params.TOL
        f_t, f_y, f_x = self.problem.d_f
        eta_t = self.solution.eta_t
        alpha_t, alpha_y = self.problem.d_alpha
        tn, h, yn = self.t[0], self.h, self.y[0]
        f, eta, alpha = self.problem.f, self.solution.eta, self.problem.alpha
        yn_plus = self.y[1]
        # WARN: theta é uma aprox de theta_i, não sei outra forma de fazer isso
        alpha_n = alpha(tn, yn)
        f_y_n = f_y(tn, yn, eta(alpha_n))
        f_x_n = f_x(tn, yn, eta(alpha_n))
        alpha_y_n = alpha_y(tn, yn)
        theta = np.squeeze((alpha_n - tn) / h)
        t2, t3 = theta**2, theta**3

        d1 = ((2/3) * t2 - (3/2) * theta + 1) * theta
        d2 = ((-2/3) * theta + 1) * t2
        d3 = ((2/3) * theta + 1) * t2
        d4 = ((2/3) * theta - 1/2) * t2

        B = np.array([[d3, d1, d1, d1], [d2, d2, d2, d2],
                      [d3, d3, d3, d3], [d4, d4, d4, d4]])

        I = np.eye(4, dtype=yn.dtype)
        # FIX: gotta make this check automatic
        if alpha_n <= tn:
            d_eta = eta_t
        else:
            d_eta = self._hat_eta_0_t

        J = I - h * np.kron(A, f_y_n + f_x_n * d_eta(alpha_n) *
                            alpha_y_n) - h * np.kron(B, f_x_n)
        lu, piv = lu_factor(J)

        def F(K):
            F = np.zeros((4, self.ndim), dtype=float)
            for i in range(4):
                ti = tn + c[i] * h
                yi = yn + c[i] * h * self.K[i-1]
                # FIX: gotta make this check automatic
                Y_tilde = self.eeta(alpha(ti, yi))
                F[i] = self.K[i] - f(ti, yi, Y_tilde)
            return F

        self.K[0:4] = [i if i != 0 else self.K[0] for i in self.K[0:4]]
        max_iter, iter = 30, 0
        diff_old, diff_new = 4, 3  # initializing stuff
        while abs((norm(diff_new)**2)/(norm(diff_old) - norm(diff_new))) >= rho * TOL and iter <= max_iter:
            # Método de Newton usando recomposição LU
            diff_old = diff_new

            FK = F(self.K[0:4])
            diff_new = lu_solve((lu, piv), - FK)
            self.K[0:4] += diff_new
            iter += 1
        if iter > max_iter:
            return False
        return True

    def _eta_0(self, theta):
        tn, h, yn = self.t[0], self.h, self.y[0]
        theta = (theta - tn) / h
        f, eta, alpha = self.problem.f, self.solution.eta, self.problem.alpha
        yn_plus = self.y[1]
        t2, t3 = theta**2, theta**3

        d1 = 2 * t3 - 3 * t2 + 1
        d2 = -2 * t3 + 3 * t2
        d3 = t3 - 2 * t2 + theta
        d4 = t3 - t2
        # if np.all(alpha(tn + h, yn_plus) <= np.full(self.ndelays, tn)):
        #     eeta = eta
        # else:
        #     eeta = self._hat_eta_0
        self.K[4] = f(tn + h, yn_plus, self.eeta(alpha(tn + h, yn_plus)))
        eta0 = d1 * yn + d2 * yn_plus + d3 * h * self.K[0] + d4 * h * self.K[4]
        return eta0

    def _hat_eta_0(self, theta):
        tn, h, yn = self.t[0], self.h, self.y[0]
        theta = (theta - tn) / h
        f, alpha = self.problem.f, self.problem.alpha
        yn_plus = self.y[1]
        t2, t3 = theta**2, theta**3

        d1 = ((2/3) * t2 - (3/2) * theta + 1) * theta
        d2 = (-(2/3) * theta + 1) * t2
        d3 = (-(2/3) * theta + 1) * t2
        d4 = ((2/3) * theta - 1/2) * t2
        eta0 = yn + h * (d1 * self.K[0] + d2 *
                         self.K[1] + d3 * self.K[2] + d4 * self.K[3])
        return eta0

    def _hat_eta_0_t(self, theta):
        tn, h, yn = self.t[0], self.h, self.y[0]
        theta = (theta - tn) / h
        f, alpha = self.problem.f, self.problem.alpha
        yn_plus = self.y[1]

        d1 = 1 - 3 * theta + 2 * theta ** 2
        d2 = -2 * (-1 + theta) * theta
        d3 = d2
        d4 = theta * (-1 + 2 * theta)
        eta0 = h * (d1 * self.K[0] + d2 *
                    self.K[1] + d3 * self.K[2] + d4 * self.K[3])
        return eta0

    def _eta_1(self, theta):
        theta1 = self.params.theta1
        theta = (theta - self.t[0]) / self.h
        t2, t3 = theta * theta, theta * theta * theta
        nom1, den1 = (theta - 1) ** 2, 2 * theta1 - 1
        yn, yn_plus = self.y[0], self.y[1]
        h = self.h
        tn, yn = self.t[0], self.y[0]
        f, eta, alpha = self.problem.f, self.solution.eta, self.problem.alpha
        d1 = nom1 * (-3 * t2 + 2 * den1 * theta + den1) / den1
        d2 = t2 * (3 * t2 - 4 * (theta1 + 1) * theta + 6 * theta1) / den1
        d3 = (
            theta
            * nom1
            * ((1 - 3 * theta1) * theta + 2 * theta1 * den1)
            / (2 * theta1 * den1)
        )
        d4 = (
            t2
            * (theta - 1)
            * ((2 - 3 * theta1) * theta + theta1 * (4 * theta1 - 3))
            / (2 * (theta1 - 1) * den1)
        )
        d5 = t2 * nom1 / (2 * theta1 * den1 * (theta1 - 1))
        tt = tn + theta1 * h
        yy = self.eta[0](tt)
        t_alpha = alpha(tt, yy)
        # if np.all(t_alpha <= np.full(self.ndelays, tn)):
        #     eeta = eta
        # else:
        #     eeta = self.eta[0]
        Y_tilde5 = self.eeta(t_alpha)
        self.K[5] = f(tt, yy, Y_tilde5)
        return (
            d1 * yn
            + d2 * yn_plus
            + d3 * h * self.K[0]
            + d4 * h * self.K[4]
            + d5 * h * self.K[5]
        )

    def _eta_1_t(self, theta):
        theta1 = self.params.theta1
        theta = (theta - self.t[0]) / self.h
        t2, t3 = theta * theta, theta * theta * theta
        nom1, den1 = (theta - 1) ** 2, 2 * theta1 - 1
        yn, yn_plus = self.y[0], self.y[1]
        h = self.h
        tn, yn = self.t[0], self.y[0]
        f, eta, alpha = self.problem.f, self.solution.eta, self.problem.alpha
        x, a = theta, theta1

        d_d1 = (12*(a - x)*(-1 + x)*x)/(-1 + 2 * a)
        d_d2 = -(12 * (a - x)*(-1 + x) * x)/(-1 + 2 * a)
        d_d3 = ((-1 + x)*(a - 6 * a * x ** 2 + x * (-1 + 2 * x) +
                (a ** 2) * (-2 + 6*x)))/(a * (-1 + 2 * a))
        d_d4 = (x*(x*(-3 + 4 * x) + a ** 2) * (-4 + 6 * x) +
                a*(3 - 6 * (x ** 2)))/((-1 + a)*(-1 + 2 * a))
        d_d5 = (x - 3 * (x ** 2) + 2 * (x ** 3)) / \
            (a - 3 * (a ** 2) + 2 * (a ** 3))

        tt = tn + theta1 * h
        alpha_tt = alpha(tt, self.eta[1](tt))
        Y_tilde5 = eta(alpha_tt)
        self.K[5] = f(tt, self.eta[0](tt), Y_tilde5)
        return (
            d_d1 * yn
            + d_d2 * yn_plus
            + d_d3 * h * self.K[0]
            + d_d4 * h * self.K[4]
            + d_d5 * h * self.K[5]
        )

    def error_est_method(self):
        # Lobatto formula now for pi1 and pi2
        f, eta, alpha = self.problem.f, self.solution.eta, self.problem.alpha

        pi1, pi2 = (5 - np.sqrt(5)) / 10, (5 + np.sqrt(5)) / 10
        t_pi1, t_pi2 = self.t[0] + pi1 * self.h, self.t[0] + pi2 * self.h
        tt1 = self.eta[1](t_pi1)
        t1 = alpha(t_pi1, tt1)
        Y_tilde6 = eta(t1)
        self.K[6] = f(t_pi1, self.eta[1](t_pi1), Y_tilde6)
        Y_tilde7 = eta(alpha(t_pi2, self.eta[1](t_pi2)))
        self.K[7] = f(t_pi2, self.eta[1](t_pi2), Y_tilde7)
        self.y_tilde = self.y[0] + self.h * (
            (1/12)*self.K[0] + (5/12) * self.K[6] +
            (5/12) * self.K[7] + (1/12) * self.K[4]
        )

    def disc_local_error_satistied(self):
        self.disc_local_error = (
            np.linalg.norm(self.y_tilde - self.y[1]) / self.h
        )  # eq 7.3.4
        # print('--------------------------------------------------------------')
        # print(
        #     f'disc_error at tn + h {self.t[0] + self.h}, tn = {self.t[0]}, h = {self.h}')
        # print('disc_local_error', self.disc_local_error)
        # input(f'real error is {self.y[1] - real_sol(self.t[0] + self.h)}')

        if self.disc_local_error <= self.params.TOL:
            return True
        else:
            self.h = self. h * (
                max(
                    self.params.omega_min,
                    min(
                        self.params.omega_max,
                        self.params.rho
                        * (self.params.TOL / self.disc_local_error) ** (1 / 4),
                    ),
                )
            )
            return False

    def uni_local_error_satistied(self):

        tn, h = self.t[0], self.h
        self.uni_local_error = (
            h * np.linalg.norm(self.eta[0](tn + (1/2) * h) - self.eta[1](tn + (1/2)*h)))
        # eq 7.3.4

        if self.disc_local_error <= self.params.TOL:
            return True
        else:
            self.h = self.h * (
                max(
                    self.params.omega_min, self.params.rho *
                    (self.params.TOL / self.disc_local_error) ** (1 / 5)
                )
            )
            return False

    def try_step_CRK(self):
        success = self.one_step_RK4()
        if not success:
            return False
        if self.overlap == False:
            eta0 = self._eta_0
        else:
            print('overlapping')
            eta0 = self._hat_eta_0
        self.eta = [self._eta_0, self._eta_1]
        self.eta_t = [self._eta_1_t, self._eta_1_t]
        self.error_est_method()
        local_disc_satisfied = self.disc_local_error_satistied()
        uni_local_disc_satistied = self.uni_local_error_satistied()

        if not local_disc_satisfied:
            print(f'failed discrete step at t = {
                  self.t[0] + self.h} with h = {self.h}')
            return False

        if not uni_local_disc_satistied:
            print(f'failed uniform step with h = {self.h}')
            return False
        # print(f'successfull step with h = {self.h}')
        # Handling divide by zero case
        if self.disc_local_error < 1e-14 or self.uni_local_error < 1e-14:
            self.h_next = self.params.omega_max * self.h
        else:
            self.h_next = self.h * max(
                self.params.omega_min,
                min(
                    self.params.omega_max,
                    self.params.rho * (self.params.TOL /
                                       self.disc_local_error)**(1/4),
                    self.params.rho * (self.params.TOL /
                                       self.uni_local_error)**(1/5)
                )
            )

        return True

    def one_step_CRK(self, max_iter=100):
        success = self.try_step_CRK()
        if success:
            return True, self
        else:
            for i in range(max_iter - 1):
                disc_found = self.is_there_disc()
                if disc_found:
                    self.solution.discs.append(disc_found)
                    # print(
                    #     '||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||')
                    # print('disc found at', self.disc)
                    # print(
                    #     '||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||')
                    self.h = float(self.disc - self.t[0])
                    # print(f'h = {self.h} new t will be {self.t[0] + self.h}')
                    # input('dnaudne')
                success = self.try_step_CRK()
                if success:
                    return True, self

            return False, 0
            print('max iterations reached')


class Problem:
    def __init__(self, f, alpha, phi, t_span, d_f=None, d_alpha=None, d_phi=None):
        self.t_span = np.array(t_span)
        self.ndim, self.ndelays, self.f, self.alpha, self.phi, self.t_span, self.d_f, self.d_alpha, self.d_phi = validade_arguments(
            f, alpha, phi, t_span, d_f, d_alpha, d_phi)


class Solution:
    def __init__(self, problem: Problem):
        self.problem = problem
        self.t = [problem.t_span[0]]
        self.y = [np.atleast_1d(problem.phi(problem.t_span[0]))]
        self.etas = [problem.phi]
        self.etas_t = [problem.d_phi]
        self.discs = [problem.t_span[0]]
        self.eta_calls = 0
        self.eta_t_calls = 0

    def update(self, t_new, y_new, interpol_new, d_interpol_new=None):
        self.t.append(t_new)
        self.y.append(y_new)
        self.etas.append(interpol_new)
        self.d_etas.append(d_interpol_new)

    # WARN: My version
    @property
    def eta(self):
        def eval(t, epsilon=1e-15):

            self.eta_calls += 1
            t = np.atleast_1d(t)  # accept scalar or array

            results = []
            for ti in t:
                idx = bisect_right(self.t, ti)

                if idx == 0:
                    if ti in self.discs:
                        results.append(self.etas[0](ti - epsilon))
                    else:
                        results.append(self.etas[0](ti))
                elif idx >= len(self.etas):
                    if ti in self.discs:
                        results.append(self.etas[-2](ti - epsilon))
                    else:
                        results.append(self.etas[-1](ti))
                elif self.t[idx - 1] <= ti <= self.t[idx]:
                    if ti in self.discs:
                        results.append(self.etas[idx - 1](ti - epsilon))
                    else:
                        results.append(self.etas[idx](ti))
                else:
                    # fallback safeguard
                    # raise ValueError('is this right?')
                    results.append(self.etas[max(0, idx - 1)](ti))

            # returns vector if t was vector, scalar if t was scalar
            return np.squeeze(results)
        return eval

    @property
    def eta_t(self):
        def eval(t, epsilon=1e-15):

            self.eta_calls += 1
            t = np.atleast_1d(t)  # accept scalar or array

            results = []
            for ti in t:
                idx = bisect_right(self.t, ti)

                if idx == 0:
                    if ti in self.discs:
                        results.append(self.etas_t[0](ti - epsilon))
                    else:
                        results.append(self.etas_t[0](ti))
                elif idx >= len(self.etas_t):
                    if ti in self.discs:
                        results.append(self.etas_t[-2](ti - epsilon))
                    else:
                        results.append(self.etas_t[-1](ti))
                elif self.t[idx - 1] <= ti <= self.t[idx]:
                    if ti in self.discs:
                        results.append(self.etas_t[idx - 1](ti - epsilon))
                    else:
                        results.append(self.etas_t[idx](ti))
                else:
                    # fallback safeguard
                    # raise ValueError('is this right?')
                    results.append(self.etas_t[max(0, idx - 1)](ti))

            # returns vector if t was vector, scalar if t was scalar
            return np.squeeze(results)
        return eval

    # WARN: this is the one before changes
    # @property
    # def eta_t(self):
    #     def eval(t):
    #         self.eta_t_calls += 1
    #         t = np.atleast_1d(t)
    #         results = []
    #         for ti in t:
    #             idx = bisect_right(self.t, ti)
    #             if idx == 0:
    #                 results.append(self.etas_t[0](ti))
    #                 continue
    #             if ti == self.t[idx - 1]:
    #                 k = idx - 1
    #                 if k < len(self.etas_t):
    #                     results.append(self.etas_t[k](ti))
    #                 else:
    #                     results.append(self.etas_t[-1](ti))
    #                 continue
    #             if idx < len(self.etas_t) and self.t[idx - 1] < ti < self.t[idx]:
    #                 results.append(self.etas_t[idx](ti))
    #             else:
    #                 results.append(self.etas_t[-1](ti))
    #         return np.squeeze(results)
    #     return eval


class Solver:

    def __init__(self, Problem):
        self.problem = Problem
        self.discs = []
        self.params = CRKParameters()
        self.eta_calls = 0
        self.ndim = 0
        self.ndelays = 0


def solve_dde(f, alpha, phi, t_span, d_f=None, d_alpha=None, d_phi=None):
    problem = Problem(f, alpha, phi, t_span, d_f, d_alpha, d_phi)
    solution = Solution(problem)
    params = CRKParameters()
    t, tf = problem.t_span
    h = (params.TOL ** (1 / 4)) * 0.1  # Initial stepsize
    # h = 0.1
    print("-" * 80)
    print("Initial h:", h)
    print("-" * 80)

    while t < tf:
        # Ensure stepsize doesn't overshoot tf
        # h = min(h, tf - t)
        if solution.t[-1] + h >= tf:
            h = tf - solution.t[-1]

        onestep = OneStep(problem, solution, h)
        success, step = onestep.one_step_CRK()
        if step.disc != False:
            solution.discs.append(step.disc)

        if success:  # Step accepted
            t = solution.t[-1] + step.h
            solution.t.append(t)
            # self.steps.append(step) #WARN: useless?
            solution.y.append(step.y[1])
            solution.etas.append(step.eta[1])
            solution.etas_t.append(step.eta_t[1])
            h = onestep.h_next  # Use adjusted stepsize from rejection

        else:
            h = onestep.h_next  # Use adjusted stepsize from rejection
            raise (f"Step rejected at t={t:.6f}, new h={h:.6e}")
    return solution
