from sympy import symbols, expand, Rational, pprint, init_printing, latex, collect, Poly, sqrt, simplify, factor, signsimp


def eta_0_stuff():
    theta = symbols('theta')

    d1 = theta - Rational(3, 2) * theta ** 2 + Rational(2, 3) * theta**3
    d2 = theta**2 - Rational(2, 3) * theta**3
    d3 = theta**2 - Rational(2, 3)*theta**3
    d4 = Rational(1, 2) * theta**2 - Rational(1, 3) * theta**3
    d5 = - theta**2 + theta**3

    theta1 = Rational(1, 3)
    print('d1:', d1.subs(theta, theta1))
    print('d2:', d2.subs(theta, theta1))
    print('d3:', d3.subs(theta, theta1))
    print('d4:', d4.subs(theta, theta1))
    print('d5:', d5.subs(theta, theta1))


def eta_1_stuff():
    theta = symbols('theta')
    theta1 = Rational(1, 3)
    b1 = (1/(2*theta1 - 1)) * (theta - 1)**2 * \
        (-3*theta**2 + 2*(2*theta1 - 1)*theta + 2*theta1 - 1)
    b2 = (1/(2*theta1 - 1)) * theta**2 * \
        (3*theta**2 - 4*(theta1 + 1)*theta + 6*theta1)
    b3 = (1/(2*theta1*(2*theta1 - 1))) * theta*(theta - 1)**2 * \
        ((1 - 3*theta1)*theta + 2*theta1*(2*theta1 - 1))
    b4 = (1/(2*(theta1 - 1)*(2*theta1 - 1))) * theta**2 * \
        (theta - 1) * ((2 - 3*theta1)*theta + theta1*(4*theta1 - 3))
    b5 = (1/(2*theta1*(2*theta1 - 1)*(theta1 - 1))) * theta**2 * (theta - 1)**2

    d1 = Rational(1, 6)*b2 + b3
    d2 = b2/3
    d3 = b2/3
    d4 = b2/6
    d5 = b4
    d6 = b5

    # Expressions for the polynomials
    # print(latex(collect(expand(d1), theta)))
    # print(latex(collect(expand(d2), theta)))
    # print(latex(collect(expand(d3), theta)))
    # print(latex(collect(expand(d4), theta)))
    # print(latex(collect(expand(d5), theta)))
    # print(latex(collect(expand(d6), theta)))

    pi1 = (5 - sqrt(5)) / 10
    pi2 = (5 + sqrt(5)) / 10

    d1_eval = simplify(d1.subs(theta, pi1))
    d2_eval = simplify(d2.subs(theta, pi1))
    d3_eval = simplify(d3.subs(theta, pi1))
    d4_eval = simplify(d4.subs(theta, pi1))
    d5_eval = simplify(d5.subs(theta, pi1))
    d6_eval = simplify(d6.subs(theta, pi1))

    print('___________________________d1_pi2__________________________________')
    print(latex(factor(d1_eval)), r', \quad')
    print(latex(factor(d2_eval)), r', \quad')
    print(latex(factor(d3_eval)), r', \quad')
    print(latex(factor(d4_eval)), r', \quad')
    print(latex(factor(d5_eval)), r', \quad')
    print(latex(factor(d6_eval)))

    d1_new_eval = simplify(d1.subs(theta, pi2))
    d2_new_eval = simplify(d2.subs(theta, pi2))
    d3_new_eval = simplify(d3.subs(theta, pi2))
    d4_new_eval = simplify(d4.subs(theta, pi2))
    d5_new_eval = simplify(d5.subs(theta, pi2))
    d6_new_eval = simplify(d6.subs(theta, pi2))

    print('___________________________d1_new_pi2__________________________________')
    print(latex(factor(d1_new_eval)), r', \quad')
    print(latex(factor(d2_new_eval)), r', \quad')
    print(latex(factor(d3_new_eval)), r', \quad')
    print(latex(factor(d4_new_eval)), r', \quad')
    print(latex(factor(d5_new_eval)), r', \quad')
    print(latex(factor(d6_new_eval)))


eta_1_stuff()
