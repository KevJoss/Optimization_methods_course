# Optimization Methods Course

**Course:** Optimization Methods — Yachay Tech University
**Professor:** Zenaida Castillo

This repository contains the code exercises, workshops, and projects developed for the Optimization Methods course. Each project lives on its own branch/folder; the code implementing the optimization methods is under [`optimization_methods/`](optimization_methods/) (see its own README for details on each file), and the experiments that exercise that code are under `experiments/`, organized one subfolder per method.

Currently, this repository holds **Project 1**: unconstrained optimization, covering the Steepest Descent, Newton, and Conjugate Gradient methods.

## Getting started

### Requirements

- Python 3.10+

### Setup

Create a virtual environment and install the dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Running an experiment

Each method has its own experiment script under `experiments/<method>/run_experiment.py`. It runs the method until convergence and saves the full iteration history to a `history.csv` file in that same folder. For example, for Steepest Descent:

```bash
python -m experiments.steepest_descent.run_experiment
```

### Viewing results

All tables and convergence plots are collected in [`results.ipynb`](results.ipynb), at the repository root, with one section per method. Open it in Jupyter or VS Code to view the already-saved outputs, or re-run it after generating new experiment data.

### Report

The LaTeX source of the written report is under [`report/`](report/).
