from rkh_refactor import *


def _simplified_Newton(self):
    time1 = time.time()
    A, b, c = self.params.A, self.params.b, self.params.c
    rho, TOL = self.params.rho, self.params.TOL
    f_t, f_y, f_x = self.problem.d_f
    eta_t = self.solution.eta_t
    alpha_t, alpha_y = self.problem.d_alpha
    tn, h, yn = self.t[0], self.h, self.y[0]
    f, eta, alpha = self.problem.f, self.eta, self.problem.alpha
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

    B = np.array([[d1, d1, d1, d1], [d2, d2, d2, d2],
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
            yi = yn + c[i] * h * K[i-1]
            Y_tilde = self.eeta(alpha(ti, yi))
            F[i] = K[i] - f(ti, yi, Y_tilde)
        return F

    self.K[0:4] = [i if i != 0 else self.K[0] for i in self.K[0:4]]

    # sol = root(F, np.squeeze(self.K[0:4]), tol=rho*TOL)
    # for i in range(4):
    #     self.K[i] = sol.x[i]
    # return True

    max_iter, iter = 30, 0
    diff_old, diff_new = 4, 3  # initializing stuff
    while abs((np.linalg.norm(diff_new)**2)/(np.linalg.norm(diff_old) - np.linalg.norm(diff_new))) >= rho * TOL and iter <= max_iter:
        # Método de Newton usando recomposição LU
        diff_old = diff_new

        FK = F(self.K[0:4])
        diff_new = lu_solve((lu, piv), - FK)
        self.K[0:4] += diff_new
        iter += 1
    if iter > max_iter:
        return False
    return True
