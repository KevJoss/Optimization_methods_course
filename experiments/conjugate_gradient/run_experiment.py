"""Run Conjugate Gradient (Fletcher-Reeves) on f1 from x0 = [1, d] until
convergence.

Saves the full iteration history (including error w.r.t. the known exact
solution) to history.csv in this same folder, for later use in results.ipynb.
"""
import csv
from pathlib import Path

import numpy as np

from optimization_methods.conjugate_gradient import conjugate_gradient
from optimization_methods.functions import D, F_STAR_F1, X_STAR_F1, f1, grad_f1

OUTPUT_DIR = Path(__file__).parent


def run():
    x0 = [1, D]
    history = conjugate_gradient(f1, grad_f1, x0, tol=1e-6, max_iter=200000)

    rows = []
    for rec in history:
        x1, x2 = rec["x"]
        error = np.linalg.norm(rec["x"] - X_STAR_F1)
        rows.append(
            {
                "k": rec["k"],
                "x1": x1,
                "x2": x2,
                "f": rec["f"],
                "grad_norm": rec["grad_norm"],
                "error": error,
            }
        )

    out_path = OUTPUT_DIR / "history.csv"
    with open(out_path, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    converged = history[-1]["grad_norm"] < 1e-6
    print(f"Converged: {converged}")
    print(f"Iterations: {len(history)}")
    print(f"Final x: {history[-1]['x']}")
    print(f"Exact x*: {X_STAR_F1}, f* = {F_STAR_F1}")
    print(f"Final f(x): {history[-1]['f']:.3e}")
    print(f"Final ||grad||: {history[-1]['grad_norm']:.3e}")
    print(f"Saved {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    run()
