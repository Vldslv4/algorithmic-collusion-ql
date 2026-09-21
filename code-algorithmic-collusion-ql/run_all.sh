#!/usr/bin/env bash
# Reproduces every experiment reported in the paper, then rebuilds the figures.
set -euo pipefail
mkdir -p results figures
Q="python qlearning_collusion.py"

echo "[1/5] baseline duopoly, 32 sessions"
$Q --n 2 --k 15 --sessions 32 --periods 1000000 --beta 1e-5 --alpha 0.15 --seed 1 \
   --out results/base_n2.json

echo "[2/5] market structure, 2 / 3 / 4 agents"
for n in 2 3 4; do
  $Q --n $n --k 11 --sessions 16 --periods 1000000 --beta 1e-5 --alpha 0.15 --seed 2 \
     --out results/struct_n$n.json
done

echo "[3/5] exploration decay"
for b in 5e-5 1e-5 4e-6; do
  $Q --n 2 --k 15 --sessions 16 --periods 1500000 --beta $b --alpha 0.15 --seed 3 \
     --out results/beta_$b.json
done

echo "[4/5] learning rate"
for a in 0.05 0.15 0.30; do
  $Q --n 2 --k 15 --sessions 16 --periods 1000000 --beta 1e-5 --alpha $a --seed 4 \
     --out results/alpha_$a.json
done

echo "[5/5] state space treatment"
$Q --n 2 --k 15 --sessions 16 --periods 1000000 --beta 1e-5 --alpha 0.15 --seed 4 \
   --state-mode own --out results/nostate_n2.json

python figures.py
python summarise.py
