"""Numerically verify the Part II stationary points found by hand.

For each function this: (1) evaluates the analytical gradient at the
claimed stationary point (should be ~[0, 0]), (2) classifies it from the
eigenvalues of the analytical Hessian there, and (3) runs Newton's Method
(reused from Part I) from a nearby starting point to check it converges
to that same point. Results are saved to results/verification_output.txt.
"""
from pathlib import Path

import numpy as np

from optimization_methods.functions import (
    X_STAR_F2A,
    X_STAR_F2B,
    f2a,
    f2b,
    grad_f2a,
    grad_f2b,
    hess_f2a,
    hess_f2b,
)
from optimization_methods.newton import newton

OUTPUT_DIR = Path(__file__).parent / "results"


def classify(hessian):
    """Classify a stationary point from the eigenvalues of the Hessian there."""
    eigenvalues = np.linalg.eigvalsh(hessian)
    if np.all(eigenvalues > 0):
        verdict = "positive definite -> local minimum"
    elif np.all(eigenvalues < 0):
        verdict = "negative definite -> local maximum"
    elif np.any(eigenvalues > 0) and np.any(eigenvalues < 0):
        verdict = "indefinite -> saddle point"
    else:
        verdict = "semidefinite (inconclusive from this test alone)"
    return verdict, eigenvalues


def verify(name, f, grad_f, hess_f, x_star, x0):
    lines = [f"=== {name} ==="]
    lines.append(f"Analytical stationary point x* = {x_star}")
    lines.append(f"grad_f(x*)                    = {grad_f(x_star)}  (expected ~[0, 0])")

    hessian = hess_f(x_star)
    verdict, eigenvalues = classify(hessian)
    lines.append(f"Hessian at x*:\n{hessian}")
    lines.append(f"Eigenvalues: {eigenvalues}")
    lines.append(f"Classification: {verdict}")

    history = newton(f, grad_f, hess_f, x0, tol=1e-10, max_iter=100)
    x_newton = history[-1]["x"]
    lines.append(f"Newton's Method from x0={x0} converged in {len(history)} iterations to: {x_newton}")
    lines.append(f"||x_newton - x*|| = {np.linalg.norm(x_newton - x_star):.3e}")
    lines.append("")
    return lines


def run():
    OUTPUT_DIR.mkdir(exist_ok=True)

    lines = []
    lines += verify(
        "Function 1: f = 3*x1 + 100/(x1*x2) + d*x2",
        f2a, grad_f2a, hess_f2a, X_STAR_F2A, x0=[4.0, 2.0],
    )
    lines += verify(
        "Function 2: f = (x1-d)^2 + x1*x2 + (x2-d)^2",
        f2b, grad_f2b, hess_f2b, X_STAR_F2B, x0=[0.0, 0.0],
    )

    text = "\n".join(lines)
    print(text)

    out_path = OUTPUT_DIR / "verification_output.txt"
    out_path.write_text(text, encoding="utf-8")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    run()
