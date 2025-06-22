import numpy as np
from scipy.optimize import root


class one_step_data:
    def __init__(self, tn, h):
        self.tn = tn
        self.h = h
        self.sol = None
        self.K1 = None 
        self.K2 = None 
        self.K3 = None 
        self.K4 = None 
        self.K5 = None 
        self.K6 = None 
        self.K7 = None 
        self.K8 = None 

class problem_data:
    def __init__(self, f, t_span, delay, eta):
        self.f = f
        self.t_span = t_span
        self.delay = delay
        self.eta = eta
        

#TODO: gotta convert all this to classes now


def RK4_one_step(f, tn, yn, h, delay, eta):
    K1 = f(tn, yn, eta(delay(tn)))
    K2 = f(tn + 0.5 * h, yn + 0.5 * h * K1 eta(np.asarray(delay(tn + 0.5 * h))))
    K3 = f(tn + 0.5 * h, yn + 0.5 * h * K2,
           eta(np.asarray(delay(tn + 0.5 * h))))
    K4 = f(tn + h, yn + h * K3, eta(np.asarray(delay(tn + h))))

    yn_plus = yn + h(K1 / 6 + K2 / 3 + K3 / 3 + K4 / 6)
    return K1, K2, K3, K4, yn_plus


def Interpolants_one_step(f, tn, yn, h, delay, eta, K1, K2, K3, K4):
    K5 = f(tn + h, yn_plus, eta(tn + h - delay(tn + h)))

    def eta_0(theta):
        t2, t3 = theta * theta, theta * theta * theta

        d1 = 2 * t3 - 3 * t2 + 1
        d2 = -2 * t3 + 3 * t2
        d3 = t3 - 2 * t2 + theta
        d4 = t3 - t2
        return d1 * yn + d2 * yn_plus + d3 * h * K1 + d4 * h * K5

    tt = tn = theta1*h
    K6 = f(tt, eta_0(tt), eta(delay(tt)))

    def eta_1(theta):
        t2, t3 = theta * theta, theta * theta * theta
        nom1, den1 = (theta - 1) ** 2, 2 * theta1 - 1
        
        d1 = nom1 * (-3 * t2 + 2 * den1 * theta + den1) / den1
        d2 = t2 * (3 * t2 - 4(theta1 + 1) * theta + 6 * theta1) / den1
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
        return d1 * yn + d2 * yn_plus + d3 * h * K1 * d4 * h * K5 + d5 * h * K6

    return eta_0, eta_1


def one_step_CRK(f, tn, yn, h, delay, eta, TOL=1e-07, theta1=1 / 3, max_rejected_steps=30, omega_min=0.5, omega_max=1.5, rho=0.1):

    max_rejected_steps = abs(int(max_rejected_steps)) + 1
    for i in range(max_rejected_steps):
        K1, K2, K3, K4, yn_plus = RK4_one_step(f, tn, yn, h, delay, eta)

        eta_0, eta_1 = lambda theta: Interpolants_one_step(
            f, tn, yn, h, delay, eta, K1, K2, K3, K4)

        # Lobatto formula now for pi1 and pi2

        pi1, pi2 = (5 - np.sqrt(5)) / 10, (5 + np.sqrt(5)) / 10
        t_pi1, t_pi2 = tn + pi1 * h, tn + pi2 * h

        K7 = f(t_pi1, eta_1(t_pi1), eta(delay(t_pi1)))
        K8 = f(t_pi2, eta_1(t_pi2), eta(delay(t_pi2)))

        yn_plus_new = yn + h * (K1 / 12 + 5 * K7 / 12 + 5 * K8 / 12 + K5 / 12)

        discrete_local_error = np.linalg.norm(
            yn_plus_new - yn_plus) / h  # eq 7.3.4

        discrete_local_error_satistied = True if discrete_error <= TOL else False

        if not discrete_local_error_satistied:
            h = max(omega_min, min(omega_max, rho *
                    (TOL/discrete_local_error)**(1/4)))*h
            continue

        max_uniform_difference = h*(32*abs(den1))*np.linalg.norm((den1/theta1)*K1 - (
            2*K2 + 2 * K3 + K4) + (3*theta1 - 2)*K5/(theta1-1) + K6/(theta1*(theta1 - 1)))

        uniform_local_error = h*max_uniform_difference

        uniform_local_error_satistied = True if uniform_local_error <= TOL else False

        if not uniform_local_error_satistied:
            h = max(omega_min, rho*(TOL/uniform_local_error)**(1/5))*h
            continue

        h_next = max(omega_min, min(omega_max, rho*(TOL/discrete_local_error)
                     ** (1/4), rho*(TOL/uniform_local_error)**(1/5)))*h

        step_sol_obj = one_step_data(tn, h, eta_1)
        next_predicted_step. h = h_next
        return step_sol_obj, next_predicted_step



def DDE_solve(f, t_span, phi, delay):

