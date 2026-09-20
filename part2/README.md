# Part II — Stationary Points

This folder covers Part II of Project 1: finding the stationary points of the two given functions by hand and classifying them with the second-order sufficiency conditions, then corroborating the findings numerically.

## Structure

- `handwritten/`: scanned pages with the hand-worked derivation (gradient, system solved, Hessian, classification) for both functions.
- `verify_stationary_points.py`: loads `f2a`/`f2b` and their analytical gradient/Hessian from `optimization_methods/functions.py`, then for each function (1) evaluates the gradient at the hand-computed stationary point, (2) classifies it from the Hessian's eigenvalues, and (3) runs Newton's Method (reused from Part I) from a nearby point to confirm it converges to that same point. Run it with:

  ```bash
  python -m part2.verify_stationary_points
  ```

- `results/`: output of the script above (`verification_output.txt`), used as the numerical evidence backing the hand-written solution.
