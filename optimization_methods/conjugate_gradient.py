"""Nonlinear Conjugate Gradient method (Fletcher-Reeves) for unconstrained
minimization."""
import numpy as np

from .line_search import backtracking_line_search


def conjugate_gradient(f, grad_f, x0, tol=1e-6, max_iter=1000):
    """Minimize f starting from x0 using the Fletcher-Reeves Conjugate
    Gradient method:

        d_0 = -grad_f(x_0)
        x_{k+1} = x_k + alpha_k * d_k
        beta_{k+1} = ||grad_f(x_{k+1})||^2 / ||grad_f(x_k)||^2
        d_{k+1} = -grad_f(x_{k+1}) + beta_{k+1} * d_k

    Instead of always moving along -grad_f(x_k) like Steepest Descent, each
    new direction d_k reuses part of the previous direction (scaled by
    beta_k), so consecutive directions are (approximately) conjugate with
    respect to the local curvature. This avoids the zig-zagging typical of
    Steepest Descent and, for exactly quadratic f, converges in at most n
    iterations (n = number of variables) under exact line search.

    Since f here is not exactly quadratic, that finite-step guarantee does
    not hold exactly, so the direction is restarted to plain steepest
    descent (beta = 0) every n iterations, which is standard practice to
    keep nonlinear CG well behaved.

    Stops when ||grad_f(x_k)|| < tol, or after max_iter iterations.

    Returns a list of per-iteration records: {k, x, f, grad_norm}.
    """
    x = np.array(x0, dtype=float)
    n = len(x)

    grad = grad_f(x)
    direction = -grad
    history = []

    for k in range(max_iter):
        grad_norm = np.linalg.norm(grad)
        history.append({"k": k, "x": x.copy(), "f": f(x), "grad_norm": grad_norm})

        if grad_norm < tol:
            break

        alpha = backtracking_line_search(f, grad_f, x, direction, grad_x=grad)
        x_new = x + alpha * direction
        grad_new = grad_f(x_new)

        restart = (k + 1) % n == 0
        beta = 0.0 if restart else (grad_new @ grad_new) / (grad @ grad)
        direction = -grad_new + beta * direction

        x, grad = x_new, grad_new

    return history
