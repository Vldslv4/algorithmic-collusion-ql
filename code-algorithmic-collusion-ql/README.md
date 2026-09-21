# Algorithmic collusion by Q-learning pricing agents

Simulation code and results for the paper *Algorithmic Collusion by Reinforcement-Learning
Pricing Agents: Simulation Evidence and Implications for Financial Markets and Competition
Law* (V. L. Dolgov, Moscow Technological Institute).

Independent Q-learning agents set prices in a repeated Bertrand oligopoly with logit demand.
The environment follows Calvano, Calzolari, Denicolò & Pastorello (2020), *American Economic
Review* 110(10), 3267-3297. The agents never communicate, never see each other's profits and
receive no information about demand. Their only feedback is the profit realised each period.

## Headline results

| Experiment | Sessions | Collusion index Δ | 95% CI | Mean price |
|---|---|---|---|---|
| Baseline duopoly, 15-price grid | 32 | 0.778 | [0.735, 0.821] | 1.764 |
| 2 agents, 11-price grid | 16 | 0.807 | [0.751, 0.863] | 1.770 |
| 3 agents, 11-price grid | 16 | 0.577 | [0.507, 0.647] | 1.705 |
| 4 agents, 11-price grid | 16 | 0.471 | [0.442, 0.500] | 1.633 |
| Own past price only in the state | 16 | −0.018 | [−0.062, 0.026] | 1.469 |

Δ = 0 is the static Bertrand-Nash outcome, Δ = 1 is full collusion. For the duopoly the
Bertrand-Nash price is 1.473 and the fully collusive price is 1.925.

Two findings drive the paper. Agents that observe the rival's previous price converge to
supra-competitive prices in every session and answer a forced deviation with a price war
followed by recovery within five to seven periods. Agents that see only their own previous
price converge to the competitive outcome, which locates the mechanism in the observation of
rival conduct rather than in reinforcement learning as such.

## Repository layout

```
qlearning_collusion.py   simulation (numpy, vectorised across sessions)
figures.py               builds the five figures from results/
run_all.sh               reproduces every experiment and figure from scratch
results/                 session-level output of each experiment (JSON)
figures/                 figures as they appear in the paper (PNG, 200 dpi)
```

## Reproducing the results

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash run_all.sh            # ~12 minutes on a single core
```

`run_all.sh` writes fresh JSON into `results/` and rebuilds `figures/`. Every run uses a fixed
seed, so the numbers in the table above reproduce exactly. Each JSON file records the full
configuration, the per-session collusion index, the average learning path and, for the baseline,
the price path around the forced deviation.

To run one configuration only:

```bash
python qlearning_collusion.py --n 2 --k 15 --sessions 32 --periods 1000000 \
       --beta 1e-5 --alpha 0.15 --seed 1 --out results/base_n2.json
```

Arguments: `--n` agents, `--k` price grid points, `--sessions` independent sessions run in
parallel, `--periods` training periods, `--alpha` learning rate, `--beta` exploration decay
rate, `--seed` random seed, `--state-mode` either `joint` (state holds all firms' previous
prices) or `own` (state holds the agent's own previous price only).

## Parameters

| Parameter | Value |
|---|---|
| Product quality index a | 2.0 |
| Outside option a₀ | 0.0 |
| Horizontal differentiation μ | 0.25 |
| Marginal cost c | 1.0 |
| Learning rate α | 0.15 (0.05 and 0.30 in robustness runs) |
| Discount factor γ | 0.95 |
| Exploration ε | exp(−βt), β = 1e-5 (5e-5 and 4e-6 in robustness runs) |
| Price grid | k points spanning [p_N − ξ(p_M − p_N), p_M + ξ(p_M − p_N)], ξ = 0.1 |
| Evaluation | 50,000 greedy periods after training, exploration off |

Both benchmarks are computed numerically rather than taken from the source: the static
equilibrium by iterating best responses, the collusive benchmark by maximising symmetric joint
profit.

## Environment

Python 3.11.15, numpy 2.4.4, matplotlib 3.10.9. Older versions of numpy (≥ 1.24) work as well.

## Citation

```
Dolgov, V. L. (2026). Algorithmic Collusion by Reinforcement-Learning Pricing Agents:
Simulation Evidence and Implications for Financial Markets and Competition Law.
Code and data: https://github.com/Vldslv4/algorithmic-collusion-ql
```

## Licence

MIT, see `LICENSE`.
