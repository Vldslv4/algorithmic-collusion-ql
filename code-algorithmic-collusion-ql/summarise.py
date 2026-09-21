"""Prints every number reported in the paper, straight from results/*.json."""
import glob
import json
import math
import os
import statistics as st

ORDER = ["base_n2", "struct_n2", "struct_n3", "struct_n4",
         "alpha_0.05", "alpha_0.15", "alpha_0.30",
         "beta_5e-5", "beta_1e-5", "beta_4e-6", "nostate_n2"]


def load(name):
    with open(os.path.join("results", name + ".json")) as f:
        return json.load(f)


def line(name, d):
    n = len(d["delta"])
    se = d["delta_sd"] / math.sqrt(n)
    lo, hi = d["delta_mean"] - 1.96 * se, d["delta_mean"] + 1.96 * se
    pi = d["pi_nash"] + d["delta_mean"] * (d["pi_mono"] - d["pi_nash"])
    uplift = 100 * (pi - d["pi_nash"]) / d["pi_nash"]
    c = d["config"]
    return (f"{name:<12} n={c['n']} k={c['k']:>2} S={n:>2} "
            f"alpha={c['alpha']:.2f} beta={c['beta']:.0e} mode={c.get('state_mode', 'joint'):<5} | "
            f"delta={d['delta_mean']:>6.3f}  95% CI=[{lo:>6.3f}, {hi:>6.3f}]  "
            f"sd={d['delta_sd']:.3f}  price={d['mean_price']:.3f}  "
            f"share>0.5={100 * d['share_above_half']:>4.0f}%  profit vs Nash={uplift:>5.1f}%")


def welch(a, b):
    ma, mb = st.mean(a), st.mean(b)
    va, vb = st.variance(a), st.variance(b)
    na, nb = len(a), len(b)
    t = (ma - mb) / math.sqrt(va / na + vb / nb)
    df = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    return t, df, ma - mb


available = {os.path.basename(p)[:-5] for p in glob.glob("results/*.json")}
for name in ORDER:
    if name in available:
        print(line(name, load(name)))

if {"alpha_0.15", "nostate_n2"} <= available:
    t, df, diff = welch(load("alpha_0.15")["delta"], load("nostate_n2")["delta"])
    print(f"\nState space treatment vs its baseline: difference {diff:.3f}, "
          f"Welch t = {t:.1f}, df = {df:.1f}")

if {"struct_n2", "struct_n4"} <= available:
    t, df, diff = welch(load("struct_n2")["delta"], load("struct_n4")["delta"])
    print(f"Two agents vs four agents:             difference {diff:.3f}, "
          f"Welch t = {t:.1f}, df = {df:.1f}")

if "base_n2" in available:
    d = load("base_n2")
    dev = d["deviation_path"]
    print("\nForced deviation, average price path (deviating agent, rival):")
    for i, row in enumerate(dev):
        print(f"  t = {i - 1:>2}   {row[0]:.4f}   {row[1]:.4f}")
