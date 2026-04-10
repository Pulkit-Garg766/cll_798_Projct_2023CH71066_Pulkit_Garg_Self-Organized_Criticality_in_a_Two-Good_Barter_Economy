# Self-Organized Criticality in a Two-Good Barter Economy

Agent-based simulation of a barter economy on a small-world network, testing
for self-organized criticality (SOC) via the statistics of death avalanches.
Course project for CLL798 (Complexity Science), IIT Delhi.

**Author:** [Pulkit Garg], [2023CH71066]
**Instructor:** Prof. Rajesh
**Date:** April 2026

## Summary

A population of N = 100 agents on a Watts-Strogatz small-world graph trades
two perishable goods (apples and oranges). Each agent must consume one of
each periodically or die; dead agents respawn after a delay. Resources are
injected via a stochastic slow drive. We measure the size distribution of
death avalanches as a function of the network's mean degree k, and look for
power-law signatures of SOC.

Motivated by Ribeiro, Vasconcelos & Cajueiro, *Physica A* 391 (2011),
"Self-organized criticality in a network of economic agents with finite
consumption."

## Files

- `soc_sim.py` — core simulation (network, agents, trade, consumption)
- `experiment.py` — connectivity sweep driver, produces all four figures
- `fig1_distribution.png` — avalanche size PDF with power-law fit
- `fig2_connectivity_sweep.png` — distributions at k = 2, 4, 8
- `fig3_summary.png` — exponent, mean/max, variance vs k
- `fig4_ccdf.png` — complementary cumulative distribution
- `report.tex` / `report.pdf` — project manuscript
- `index.html` — interactive pedagogical demo of the agent logic (optional)

## Requirements

```
python >= 3.9
numpy
networkx
matplotlib
```

Install with:
```bash
pip install numpy networkx matplotlib
```

## Reproducing the results

```bash
python experiment.py
```

This runs the full sweep (3 values of k × 200 independent runs × 10,000 ticks
at N = 100) and writes the four figures to the current directory. Runtime is
approximately [X minutes] on a modern laptop.

## Key parameters (editable at the top of `experiment.py`)

| Parameter | Value | Description |
|-----------|-------|-------------|
| N | 100 | Number of agents |
| T | 10000 | Ticks per run |
| RUNS | 200 | Independent runs per k |
| K_VALUES | [2, 4, 8] | Mean degrees swept |
| WINDOW | 1 | Avalanche grouping window |

## Key results

| k | α (exponent) | mean size | max size | variance |
|---|--------------|-----------|----------|----------|
| 2 | 2.25 | 2.3 | ~95 | 38 |
| 4 | 2.21 | 2.3 | ~90 | 39 |
| 8 | 2.19 | 2.4 | ~95 | 42 |

The avalanche-size distribution shows approximately power-law behaviour
over one decade. The exponent α decreases monotonically with k, and the
variance grows with k — both consistent with a system approaching a
critical state. Full discussion in `report.pdf`.

## Acknowledgments

AI assistance (Claude, Anthropic) was used for simulation development,
debugging, and draft feedback. All modelling choices, parameter selection,
and interpretation are the author's own. The full list of prompts used is
provided in the appendix of the report.

## License

[MIT / or whatever you prefer]
