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


# f1 is a sum of two squared (even-power) terms, so f1 >= 0 everywhere.
# The minimum f1 = 0 is reached when both terms vanish: x1 = d and x1 = 2*d*x2,
# i.e. x2 = 1/2. This exact solution is used as reference to measure error.
X_STAR_F1 = np.array([D, 0.5])
F_STAR_F1 = 0.0
