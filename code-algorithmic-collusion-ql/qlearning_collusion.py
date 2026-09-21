"""
Q-learning pricing agents in a Bertrand oligopoly with logit demand.
Replication-style experiment following Calvano, Calzolari, Denicolo & Pastorello (2020).

Vectorised across independent sessions with numpy.
"""
import numpy as np
import json
import time
import argparse


# ----------------------------------------------------------------------------
# Economic environment
# ----------------------------------------------------------------------------

A_QUALITY = 2.0
A_OUTSIDE = 0.0
MU = 0.25
COST = 1.0


def quantities(prices):
    """prices: array (..., n) -> logit demand shares."""
    e = np.exp((A_QUALITY - prices) / MU)
    return e / (e.sum(axis=-1, keepdims=True) + np.exp(A_OUTSIDE / MU))


def profits_from_prices(prices):
    return (prices - COST) * quantities(prices)


def best_response(p_others, n, fine=np.linspace(1.0, 2.6, 32001)):
    """Best response of firm i when all rivals charge p_others."""
    grid = np.tile(p_others, (fine.size, n))
    grid[:, 0] = fine
    pi = profits_from_prices(grid)[:, 0]
    return fine[np.argmax(pi)]


def nash_price(n, iters=400):
    p = 1.5
    for _ in range(iters):
        p_new = best_response(p, n)
        if abs(p_new - p) < 1e-9:
            break
        p = 0.5 * p + 0.5 * p_new
    return p


def monopoly_price(n, fine=np.linspace(1.0, 3.0, 40001)):
    """Symmetric fully collusive price (joint profit maximum)."""
    grid = np.tile(fine[:, None], (1, n))
    pi = profits_from_prices(grid)[:, 0]
    return fine[np.argmax(pi)]


def build_profit_table(price_grid, n):
    """Profit tensor of shape (k,)*n + (n,)."""
    k = price_grid.size
    mesh = np.meshgrid(*([price_grid] * n), indexing="ij")
    prices = np.stack(mesh, axis=-1)          # (k,)*n + (n,)
    return profits_from_prices(prices).astype(np.float64)


# ----------------------------------------------------------------------------
# Q-learning experiment
# ----------------------------------------------------------------------------

def run_experiment(n=2, k=15, sessions=24, periods=1_000_000, alpha=0.15,
                   beta=1e-5, gamma=0.95, xi=0.1, seed=0, eval_periods=50_000,
                   track_every=2000, state_mode="joint"):
    rng = np.random.default_rng(seed)

    p_nash = nash_price(n)
    p_mono = monopoly_price(n)
    pi_nash = profits_from_prices(np.full(n, p_nash))[0]
    pi_mono = profits_from_prices(np.full(n, p_mono))[0]

    price_grid = np.linspace(p_nash - xi * (p_mono - p_nash),
                             p_mono + xi * (p_mono - p_nash), k)
    profit_tab = build_profit_table(price_grid, n)          # (k,)*n + (n,)
    profit_flat = profit_tab.reshape(-1, n)                 # (k**n, n)

    n_states = k ** n
    # profit tensor is built with indexing="ij", so firm 0 is the slowest axis
    powers = np.array([k ** (n - 1 - i) for i in range(n)], dtype=np.int64)

    S = sessions
    Q = np.zeros((S, n, n_states, k), dtype=np.float32)
    # optimistic initialisation at discounted average payoff (Calvano et al.)
    Q += (profit_tab.mean(axis=tuple(range(n))).mean() / (1 - gamma)).astype(np.float32)

    actions = rng.integers(0, k, size=(S, n))

    def to_states(acts):
        if state_mode == "joint":
            j = (acts * powers).sum(axis=1)
            return np.repeat(j[:, None], n, axis=1)
        return acts.copy()

    states = to_states(actions)

    sess_idx = np.arange(S)[:, None]
    firm_idx = np.arange(n)[None, :]

    price_track = []
    t0 = time.time()

    for t in range(periods):
        eps = np.exp(-beta * t)
        q_sa = Q[sess_idx, firm_idx, states, :]                # (S, n, k)
        greedy = q_sa.argmax(axis=2)
        explore = rng.random((S, n)) < eps
        rand_a = rng.integers(0, k, size=(S, n))
        actions = np.where(explore, rand_a, greedy)

        joint = (actions * powers).sum(axis=1)
        rewards = profit_flat[joint]                            # (S, n)

        next_states = to_states(actions)
        max_next = Q[sess_idx, firm_idx, next_states, :].max(axis=2)
        old = Q[sess_idx, firm_idx, states, actions]
        Q[sess_idx, firm_idx, states, actions] = (
            (1 - alpha) * old + alpha * (rewards + gamma * max_next)
        ).astype(np.float32)

        states = next_states

        if t % track_every == 0:
            price_track.append([t, float(price_grid[actions].mean())])

    train_time = time.time() - t0

    # ---- greedy evaluation (no exploration) --------------------------------
    ev_prices = np.zeros((S, n))
    ev_profits = np.zeros((S, n))
    for t in range(eval_periods):
        q_sa = Q[sess_idx, firm_idx, states, :]
        actions = q_sa.argmax(axis=2)
        joint = (actions * powers).sum(axis=1)
        ev_prices += price_grid[actions]
        ev_profits += profit_flat[joint]
        states = to_states(actions)
    ev_prices /= eval_periods
    ev_profits /= eval_periods

    delta = (ev_profits.mean(axis=1) - pi_nash) / (pi_mono - pi_nash)

    # ---- deviation / punishment experiment ---------------------------------
    dev_path = None
    if n == 2 and state_mode == "joint":
        horizon = 15
        paths = np.zeros((S, horizon + 1, n))
        st = states.copy()
        # settle into the stationary greedy path
        for _ in range(100):
            a = Q[sess_idx, firm_idx, st, :].argmax(axis=2)
            st = to_states(a)
        a_eq = Q[sess_idx, firm_idx, st, :].argmax(axis=2)
        paths[:, 0, :] = price_grid[a_eq]
        # forced one-shot deviation of firm 0 to the Nash-price grid point
        nash_cell = int(np.argmin(np.abs(price_grid - p_nash)))
        a_dev = a_eq.copy()
        a_dev[:, 0] = nash_cell
        paths[:, 1, :] = price_grid[a_dev]
        st = to_states(a_dev)
        for h in range(2, horizon + 1):
            a = Q[sess_idx, firm_idx, st, :].argmax(axis=2)
            paths[:, h, :] = price_grid[a]
            st = to_states(a)
        dev_path = paths.mean(axis=0).tolist()

    out = {
        "config": dict(n=n, k=k, sessions=S, periods=periods, alpha=alpha,
                       beta=beta, gamma=gamma, xi=xi, seed=seed,
                       state_mode=state_mode),
        "p_nash": float(p_nash), "p_mono": float(p_mono),
        "pi_nash": float(pi_nash), "pi_mono": float(pi_mono),
        "price_grid": price_grid.tolist(),
        "delta": delta.tolist(),
        "delta_mean": float(delta.mean()),
        "delta_median": float(np.median(delta)),
        "delta_sd": float(delta.std(ddof=1)),
        "delta_min": float(delta.min()),
        "delta_max": float(delta.max()),
        "share_above_half": float((delta > 0.5).mean()),
        "mean_price": float(ev_prices.mean()),
        "mean_price_by_session": ev_prices.mean(axis=1).tolist(),
        "price_track": price_track,
        "deviation_path": dev_path,
        "train_seconds": train_time,
    }
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--sessions", type=int, default=24)
    ap.add_argument("--periods", type=int, default=1_000_000)
    ap.add_argument("--beta", type=float, default=1e-5)
    ap.add_argument("--alpha", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--state-mode", type=str, default="joint")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    res = run_experiment(n=args.n, k=args.k, sessions=args.sessions,
                         periods=args.periods, alpha=args.alpha,
                         beta=args.beta, seed=args.seed,
                         state_mode=args.state_mode)
    with open(args.out, "w") as f:
        json.dump(res, f)
    print(json.dumps({kk: vv for kk, vv in res.items()
                      if kk in ("delta_mean", "delta_sd", "delta_min", "delta_max",
                                "share_above_half", "mean_price", "p_nash", "p_mono",
                                "train_seconds")}, indent=2))
