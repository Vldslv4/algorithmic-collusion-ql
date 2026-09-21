"""Builds the five figures reported in the paper from results/*.json."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE = "#1a5f9e", "#d1671f"
GRAY, INK = "#5a5a5a", "#1f1f1f"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.edgecolor": "#9a9a9a", "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": "#4a4a4a", "ytick.color": "#4a4a4a",
    "axes.grid": True, "grid.color": "#dcdcdc", "grid.linewidth": 0.6,
    "figure.dpi": 200, "savefig.dpi": 200, "savefig.bbox": "tight",
})


def clean(ax, grid_axis="y"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis=grid_axis, alpha=0.8)
    ax.set_axisbelow(True)


def ci95(d):
    return 1.96 * d["delta_sd"] / np.sqrt(len(d["delta"]))


base = json.load(open("results/base_n2.json"))
p_nash, p_mono = base["p_nash"], base["p_mono"]

# Figure 1: learning path -----------------------------------------------------
track = np.array(base["price_track"])
w = 12
smooth = np.convolve(track[:, 1], np.ones(w) / w, mode="valid")
x = track[w - 1:, 0] / 1e6

fig, ax = plt.subplots(figsize=(6.4, 3.5))
ax.plot(x, smooth, color=BLUE, lw=2)
ax.axhline(p_mono, color=GRAY, ls="--", lw=1.2)
ax.axhline(p_nash, color=GRAY, ls=":", lw=1.2)
ax.text(x[-1], p_mono + 0.006, f"Monopoly price {p_mono:.3f}", ha="right",
        va="bottom", fontsize=9, color=GRAY)
ax.text(x[-1], p_nash - 0.008, f"Bertrand-Nash price {p_nash:.3f}", ha="right",
        va="top", fontsize=9, color=GRAY)
ax.set_xlabel("Training periods, millions")
ax.set_ylabel("Average posted price")
ax.set_ylim(1.44, 1.96)
clean(ax)
fig.savefig("figures/fig_en_1_learning.png")
plt.close(fig)

# Figure 2: distribution of the index -----------------------------------------
delta = np.array(base["delta"])
fig, ax = plt.subplots(figsize=(6.4, 3.4))
ax.hist(delta, bins=np.arange(0.4, 1.05, 0.05), color=BLUE,
        edgecolor="white", linewidth=1.2)
ax.axvline(delta.mean(), color=ORANGE, lw=2)
ax.text(delta.mean() + 0.012, ax.get_ylim()[1] * 0.92,
        f"mean $\\Delta$ = {delta.mean():.3f}", color=ORANGE, fontsize=9.5)
ax.set_xlabel("Collusion index $\\Delta$")
ax.set_ylabel("Number of sessions")
clean(ax)
fig.savefig("figures/fig_en_2_delta_dist.png")
plt.close(fig)

# Figure 3: market structure --------------------------------------------------
ns, means, errs = [], [], []
for n in (2, 3, 4):
    d = json.load(open(f"results/struct_n{n}.json"))
    ns.append(n)
    means.append(d["delta_mean"])
    errs.append(ci95(d))

fig, ax = plt.subplots(figsize=(6.0, 3.4))
xpos = np.arange(len(ns))
ax.bar(xpos, means, width=0.5, color=BLUE, yerr=errs, capsize=4,
       error_kw=dict(ecolor=GRAY, lw=1.2))
for xi, m, e in zip(xpos, means, errs):
    ax.text(xi, m + e + 0.04, f"{m:.2f}", ha="center", fontsize=10, color=INK)
ax.set_xticks(xpos)
ax.set_xticklabels([f"{n} firms" for n in ns])
ax.set_ylabel("Collusion index $\\Delta$")
ax.set_ylim(0, 1.0)
clean(ax)
fig.savefig("figures/fig_en_3_nfirms.png")
plt.close(fig)

# Figure 4: deviation and punishment ------------------------------------------
dev = np.array(base["deviation_path"])
t = np.arange(-1, len(dev) - 1)
fig, ax = plt.subplots(figsize=(6.4, 3.6))
ax.plot(t, dev[:, 0], color=ORANGE, lw=2, marker="o", ms=4.5, label="Deviating agent")
ax.plot(t, dev[:, 1], color=BLUE, lw=2, marker="s", ms=4.5, label="Rival agent")
ax.axhline(p_nash, color=GRAY, ls=":", lw=1.2)
ax.text(t[0], p_nash - 0.006, "Bertrand-Nash price", ha="left", va="top",
        fontsize=9, color=GRAY)
ax.axvline(0, color=GRAY, lw=0.9, alpha=0.7)
ax.text(0.15, 1.90, "forced deviation", fontsize=9, color=GRAY)
ax.set_xlabel("Periods relative to the deviation")
ax.set_ylabel("Average posted price")
ax.set_ylim(1.42, 1.94)
ax.legend(frameon=False, loc="lower right", fontsize=9.5)
clean(ax)
fig.savefig("figures/fig_en_4_deviation.png")
plt.close(fig)

# Figure 5: state space treatment ---------------------------------------------
a15 = json.load(open("results/alpha_0.15.json"))
nost = json.load(open("results/nostate_n2.json"))
groups = [("Rival's past price\nin the state space", np.array(a15["delta"]), BLUE),
          ("Own past price only", np.array(nost["delta"]), ORANGE)]

fig, ax = plt.subplots(figsize=(6.0, 3.6))
rng = np.random.default_rng(7)
for i, (lab, vals, col) in enumerate(groups):
    ax.bar(i, vals.mean(), width=0.45, color=col, alpha=0.85,
           edgecolor="white", linewidth=0.8)
    ax.scatter(i + rng.uniform(-0.13, 0.13, vals.size), vals, s=16,
               color="white", edgecolor=GRAY, linewidth=0.8, zorder=3)
    ax.text(i, vals.max() + 0.07, f"mean {vals.mean():.3f}", ha="center",
            fontsize=10.5, color=INK)
ax.axhline(0, color=GRAY, lw=1)
ax.set_xticks([0, 1])
ax.set_xticklabels([g[0] for g in groups], fontsize=9.5)
ax.set_ylabel("Collusion index $\\Delta$")
ax.set_ylim(-0.2, 1.05)
clean(ax)
fig.savefig("figures/fig_en_5_statespace.png")
plt.close(fig)

print("figures written to figures/")
