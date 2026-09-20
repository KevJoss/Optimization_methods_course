"""Test functions used in Project 1.

D is the personal parameter: last digit of the student ID (1) plus 2.
"""
import numpy as np

D = 3


def f1(x, d=D):
    """Objective function of Part I: f(x1, x2) = (x1 - d)^4 + (x1 - 2*d*x2)^2."""
    x1, x2 = x
    return (x1 - d) ** 4 + (x1 - 2 * d * x2) ** 2


def grad_f1(x, d=D):
    """Gradient of f1, obtained analytically via the chain rule."""
    x1, x2 = x
    term = x1 - 2 * d * x2
    df_dx1 = 4 * (x1 - d) ** 3 + 2 * term
    df_dx2 = -4 * d * term
    return np.array([df_dx1, df_dx2])


def hess_f1(x, d=D):
    """Hessian of f1, obtained analytically by differentiating grad_f1 again."""
    x1, x2 = x
    d2f_dx1dx1 = 12 * (x1 - d) ** 2 + 2
    d2f_dx1dx2 = -4 * d
    d2f_dx2dx2 = 8 * d ** 2
    return np.array([[d2f_dx1dx1, d2f_dx1dx2], [d2f_dx1dx2, d2f_dx2dx2]])


# f1 is a sum of two squared (even-power) terms, so f1 >= 0 everywhere.
# The minimum f1 = 0 is reached when both terms vanish: x1 = d and x1 = 2*d*x2,
# i.e. x2 = 1/2. This exact solution is used as reference to measure error.
X_STAR_F1 = np.array([D, 0.5])
F_STAR_F1 = 0.0


def f2a(x, d=D):
    """Part II, function 1: f(x1, x2) = 3*x1 + 100/(x1*x2) + d*x2. Domain: x1, x2 != 0."""
    x1, x2 = x
    return 3 * x1 + 100 / (x1 * x2) + d * x2


def grad_f2a(x, d=D):
    """Gradient of f2a."""
    x1, x2 = x
    df_dx1 = 3 - 100 / (x1 ** 2 * x2)
    df_dx2 = d - 100 / (x1 * x2 ** 2)
    return np.array([df_dx1, df_dx2])


def hess_f2a(x, d=D):
    """Hessian of f2a."""
    x1, x2 = x
    d2f_dx1dx1 = 200 / (x1 ** 3 * x2)
    d2f_dx1dx2 = 100 / (x1 ** 2 * x2 ** 2)
    d2f_dx2dx2 = 200 / (x1 * x2 ** 3)
    return np.array([[d2f_dx1dx1, d2f_dx1dx2], [d2f_dx1dx2, d2f_dx2dx2]])


# Setting grad_f2a = 0 gives x1^2*x2 = 100/3 and x1*x2^2 = 100/d. For this
# project's d = 3 both right-hand sides coincide, forcing x1 = x2 = (100/3)^(1/3).
# (For a general d != 3 this stationary point would not have x1 = x2.)
X_STAR_F2A = np.array([(100 / 3) ** (1 / 3), (100 / 3) ** (1 / 3)])


def f2b(x, d=D):
    """Part II, function 2: f(x1, x2) = (x1-d)^2 + x1*x2 + (x2-d)^2."""
    x1, x2 = x
    return (x1 - d) ** 2 + x1 * x2 + (x2 - d) ** 2


def grad_f2b(x, d=D):
    """Gradient of f2b."""
    x1, x2 = x
    df_dx1 = 2 * (x1 - d) + x2
    df_dx2 = x1 + 2 * (x2 - d)
    return np.array([df_dx1, df_dx2])


def hess_f2b(x, d=D):
    """Hessian of f2b (constant, since f2b is quadratic)."""
    return np.array([[2.0, 1.0], [1.0, 2.0]])


# Setting grad_f2b = 0 gives 2*x1+x2 = 2*d and x1+2*x2 = 2*d, whose unique
# solution is x1 = x2 = 2*d/3.
X_STAR_F2B = np.array([2 * D / 3, 2 * D / 3])
