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

## Reference Results

The following values are the reported simulation results from the paper.

### Multi-task TMC Paths

| Task | Path | State Transition | Type | Probability | Expected Correctness |
|---|---:|---|---|---:|---:|
| TASK1 | 0 | `S1[0] -> S2[0] -> S3[0] -> S4[0]` | **Easy** | 0.413223 | 0.727995 |
| TASK1 | 1 | `S1[0] -> S2[0] -> S3[1] -> S4[0]` | Hard | 0.075131 | 0.132363 |
| TASK1 | 2 | `S1[0] -> S2[1] -> S3[0] -> S4[0]` | Hard | 0.004132 | 0.007280 |
| TASK1 | 3 | `S1[0] -> S2[1] -> S3[1] -> S4[0]` | Hard | 0.075131 | 0.132363 |
| TASK2 | 0 | `S1[1] -> S2[0] -> S3[0] -> S4[1]` | **Easy** | 0.413223 | 0.955691 |
| TASK2 | 1 | `S1[1] -> S2[0] -> S3[1] -> S4[1]` | Hard | 0.007513 | 0.017376 |
| TASK2 | 2 | `S1[1] -> S2[1] -> S3[0] -> S4[1]` | Hard | 0.004132 | 0.009557 |
| TASK2 | 3 | `S1[1] -> S2[1] -> S3[1] -> S4[1]` | Hard | 0.007513 | 0.017376 |

### CoT Generation Statistics

| Strategy | TASK1 Easy | TASK1 Hard | TASK1 Invalid | TASK2 Easy | TASK2 Hard | TASK2 Invalid |
|---|---:|---:|---:|---:|---:|---:|
| Base Model | 21.67% | 8.07% | 70.27% | 20.03% | 1.10% | 78.87% |
| REINFORCE | **94.33%** | 3.43% | 2.23% | **1.52%** | 0.87% | 97.62% |
| RAFT | **95.22%** | 2.33% | 2.45% | **2.30%** | 0.92% | 96.78% |
| PPO | **91.82%** | 5.40% | 2.78% | **2.23%** | 1.03% | 96.73% |
| RL-rej | 49.62% | **17.42%** | 32.97% | 30.63% | 2.27% | 67.10% |
| GRPO-KL | 46.47% | **16.27%** | 37.27% | 54.18% | 1.68% | 44.13% |
| Soft-BoN | 8.98% | **19.30%** | 71.72% | 7.00% | 17.27% | 75.73% |
| ORM-BoN | 21.00% | 7.30% | 71.70% | 20.23% | 0.97% | 78.80% |
| PRM-BoN | 99.13% | 0.87% | 0.00% | 13.42% | 36.77% | 49.82% |
| DPRM-BoN | 99.52% | 0.48% | 0.00% | 13.40% | 37.02% | 49.58% |
| DPRM-AS | 17.23% | **36.10%** | 46.67% | 12.02% | 38.02% | 49.97% |

## Citation

```bibtex
@misc{bu2026distributionalbiasesposttraining,
  title        = {Distributional Biases in Post-Training: A Markovian Analysis of Reasoning Trajectories},
  author       = {Dake Bu and Wei Huang and Andi Han and Bo Xue and Hau-San Wong and Qingfu Zhang and Taiji Suzuki and Atsushi Nitanda},
  year         = {2026},
  url          = {https://openreview.net/pdf?id=3dPpfbmZ3n}
}
```
