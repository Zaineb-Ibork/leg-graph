"""E4: how the value of inter-leg information grows as the target leg's own signal gets staler.

Usage:
    python delay_sweep.py --data data --out results
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from legraph import Normalizer, load_episodes, make_samples
from train import MODELS, hold_last, predict, rmse_deg, train_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data")
    p.add_argument("--out", default="results")
    p.add_argument("--delays", type=int, nargs="+", default=[5, 10, 20, 40])  # samples (2 ms each)
    p.add_argument("--window", type=int, default=10)
    p.add_argument("--epochs", type=int, default=8)
    a = p.parse_args()
    tr_eps, te_eps = load_episodes(a.data, seeds=[0, 1, 2, 3]), load_episodes(a.data, seeds=[4, 5])
    norm = Normalizer(tr_eps)
    res = {}
    for d in a.delays:
        tr = make_samples(tr_eps, norm, a.window, d)
        X, M, Y, _ = make_samples(te_eps, norm, a.window, d)
        r = {"hold last value": rmse_deg(hold_last(X, M, a.window), Y, norm, M)}
        for name, kind in MODELS.items():
            r[name] = rmse_deg(predict(train_model(kind, tr, a.epochs), X, M), Y, norm, M)
        res[d * 2] = r
        print(f"delay {d*2} ms", {k: round(v, 2) for k, v in r.items()})
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "delay_sweep.json").write_text(json.dumps(res, indent=2))
    fig, ax = plt.subplots(figsize=(6, 3.4), tight_layout=True)
    ms = sorted(res)
    for name in ["own history only", "anatomical leg graph", "fully connected + attention"]:
        ax.plot(ms, [res[m][name] for m in ms], "o-", label=name)
    ax.set_xlabel("delay of the hidden leg's own signal (ms)")
    ax.set_ylabel("RMSE (deg)")
    ax.set_title("Value of the other legs grows with delay (step cycle ~83 ms)")
    ax.legend(fontsize=8, frameon=False)
    fig.savefig(out / "e4_delay_sweep.png", dpi=160)


if __name__ == "__main__":
    main()
