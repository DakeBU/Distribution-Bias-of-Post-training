# Distributional Biases in Post-Training

Code for the synthetic Multi-task Tree-structured Markov Chain (TMC) experiments in:

**Distributional Biases in Post-Training: A Markovian Analysis of Reasoning Trajectories**  
OpenReview: https://openreview.net/pdf?id=3dPpfbmZ3n

This release contains only the abstract Multi-task TMC simulation used for Section 5 and Appendix C of the paper.

## Contents

```
multi_task_tmc.py   # TMC construction, RLVR-style fine-tuning, inference scaling, and plotting
requirements.txt    # Minimal Python dependencies
LICENSE             # Apache-2.0
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If `python3 -m venv` is unavailable, install the dependencies in any existing Python environment:

```bash
pip install -r requirements.txt
```

## Reproduce the Simulation

```bash
python multi_task_tmc.py
```

The script prints the pass@30 and valid-CoT coverage tables, then writes Appendix C style plots to `figures/`:

- `TASK1_performance.png`
- `TASK2_performance.png`
- `TASK1_coverage.png`
- `TASK2_coverage.png`

Default parameters match the paper simulation: `L=4`, `M0=2`, `M=2`, `N=15`, `K=30`, pretraining steps `T1=2000`, `T2=500`, and fine-tuning steps `T=1000`.

For a fast smoke test:

```bash
python multi_task_tmc.py --quick
```
