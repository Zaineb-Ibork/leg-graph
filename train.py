"""Leg imputation experiments on NeuroMechFly walking data.

Question: how much does the *current* state of the other legs tell us about the
current posture of a leg whose own signal is only available with a delay?
If inter-leg coordination is strong, a graph network passing messages between
legs should recover the hidden leg far better than extrapolating its own past.

Experiments
  E1  within-gait: train on seeds 0-3, test on seeds 4-5 (all gaits)
  E2  cross-gait:  train on tripod + tetrapod, test on wave (unseen coordination)
  E3  attention:   which legs the fully connected model attends to, per gait

Usage:
    python train.py --data data --out results
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from legraph import LEGS, N_DOF, LegGraphNet, Normalizer, adjacency, load_episodes, make_samples

MODELS = {"own history only": "none", "anatomical leg graph": "anatomical", "fully connected + attention": "full"}


def rmse_deg(pred, y, norm, m):
    sd = m @ torch.tensor(norm.sd, dtype=torch.float32)       # scale of the target leg
    err = (pred - y) * sd
    return float(torch.sqrt((err ** 2).mean()) * 180 / np.pi)


def train_model(kind, data, epochs, seed=0):
    torch.manual_seed(seed)
    X, M, Y, _ = data
    model = LegGraphNet(X.shape[-1], adjacency(kind))
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for ep in range(epochs):
        perm = torch.randperm(len(X))
        model.train()
        for i in range(0, len(X), 512):
            b = perm[i:i + 512]
            loss = torch.nn.functional.mse_loss(model(X[b], M[b]), Y[b])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    return model.eval()


@torch.no_grad()
def predict(model, X, M):
    return torch.cat([model(X[i:i + 4096], M[i:i + 4096]) for i in range(0, len(X), 4096)])


def hold_last(X, M, window):
    """Baseline: repeat the last (delayed) observation of the target leg."""
    feat = X.reshape(*X.shape[:2], window, -1)                 # (B, 6, W, 8)
    last = feat[:, :, -1, :N_DOF]
    return (last * M[..., None]).sum(1)


def per_gait(fn, G):
    return {g: fn(G == g) for g in sorted(set(G))}


@torch.no_grad()
def attention_maps(model, X, M, G):
    """Last-layer attention of the target leg's row, averaged per gait -> (6, 6)."""
    maps = {}
    for g in sorted(set(G)):
        idx = np.where(G == g)[0]
        A = torch.zeros(6, 6)
        for k in range(6):
            sel = idx[M[idx, k].numpy() == 1][:3000]
            model(X[sel], M[sel])
            A[k] = model.attention()[-1][:, k].mean(0)
        maps[g] = A.numpy()
    return maps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data")
    p.add_argument("--out", default="results")
    p.add_argument("--window", type=int, default=10)   # 20 ms at 500 Hz
    p.add_argument("--delay", type=int, default=10)    # target leg seen 20 ms late
    p.add_argument("--epochs", type=int, default=12)
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    res = {"window_ms": a.window * 2, "delay_ms": a.delay * 2}

    # ---------------- E1 within-gait, held-out seeds
    tr_eps = load_episodes(a.data, seeds=[0, 1, 2, 3])
    te_eps = load_episodes(a.data, seeds=[4, 5])
    norm = Normalizer(tr_eps)
    tr = make_samples(tr_eps, norm, a.window, a.delay)
    te = make_samples(te_eps, norm, a.window, a.delay)
    Xte, Mte, Yte, Gte = te
    e1 = {"hold last value": per_gait(lambda s: rmse_deg(hold_last(Xte[s], Mte[s], a.window), Yte[s], norm, Mte[s]), Gte)}
    models = {}
    for name, kind in MODELS.items():
        models[name] = train_model(kind, tr, a.epochs)
        P = predict(models[name], Xte, Mte)
        e1[name] = per_gait(lambda s: rmse_deg(P[s], Yte[s], norm, Mte[s]), Gte)
        print("E1", name, e1[name])
    res["E1_within_gait_rmse_deg"] = e1

    # ---------------- E2 cross-gait: train tripod+tetrapod, test wave
    tr2_eps = load_episodes(a.data, gaits=["tripod", "tetrapod"])
    te2_eps = load_episodes(a.data, gaits=["wave"])
    norm2 = Normalizer(tr2_eps)
    tr2 = make_samples(tr2_eps, norm2, a.window, a.delay)
    X2, M2, Y2, _ = make_samples(te2_eps, norm2, a.window, a.delay)
    e2 = {"hold last value": rmse_deg(hold_last(X2, M2, a.window), Y2, norm2, M2)}
    for name, kind in MODELS.items():
        e2[name] = rmse_deg(predict(train_model(kind, tr2, a.epochs), X2, M2), Y2, norm2, M2)
        print("E2", name, e2[name])
    res["E2_cross_gait_wave_rmse_deg"] = e2

    # ---------------- E3 attention maps of the fully connected model
    maps = attention_maps(models["fully connected + attention"], Xte, Mte, Gte)
    res["E3_attention"] = {g: np.round(m, 3).tolist() for g, m in maps.items()}
    (out / "results.json").write_text(json.dumps(res, indent=2))

    # ---------------- figures
    gaits = sorted(set(Gte))
    names = list(e1)
    fig, ax = plt.subplots(figsize=(7.5, 3.4), tight_layout=True)
    w = 0.8 / len(names)
    for i, n in enumerate(names):
        ax.bar(np.arange(len(gaits)) + i * w, [e1[n][g] for g in gaits], w, label=n)
    ax.set_xticks(np.arange(len(gaits)) + w * (len(names) - 1) / 2, gaits)
    ax.set_ylabel("RMSE of hidden leg (deg)")
    ax.set_title(f"Imputing a leg seen {a.delay*2} ms late, held-out seeds")
    ax.legend(fontsize=8, frameon=False)
    fig.savefig(out / "e1_imputation_rmse.png", dpi=160)

    fig, axs = plt.subplots(1, len(gaits), figsize=(3.4 * len(gaits), 3.1), constrained_layout=True)
    for ax, g in zip(axs, gaits):
        im = ax.imshow(maps[g], vmin=0, vmax=maps[g].max(), cmap="magma")
        ax.set_xticks(range(6), LEGS); ax.set_yticks(range(6), LEGS)
        ax.set_xlabel("attended leg"); ax.set_title(g)
    axs[0].set_ylabel("hidden (target) leg")
    fig.colorbar(im, ax=axs, shrink=0.8)
    fig.savefig(out / "e3_attention_maps.png", dpi=160)

    # example trace: femur-tibia joint of the left middle leg, tripod test episode
    m = models["fully connected + attention"]
    sel = np.where((Gte == "tripod") & (Mte[:, 1].numpy() == 1))[0][:300]
    P = predict(m, Xte[sel], Mte[sel]); H = hold_last(Xte[sel], Mte[sel], a.window)
    sd = norm.sd[1, 5]; mu = norm.mu[1, 5]
    t = np.arange(len(sel)) * 2
    fig, ax = plt.subplots(figsize=(7.5, 2.8), tight_layout=True)
    for v, lab, st in [(Yte[sel, 5], "ground truth", "k-"), (P[:, 5], "graph model", "C2-"), (H[:, 5], "hold last value", "C0--")]:
        ax.plot(t, np.degrees(v.numpy() * sd + mu), st, lw=1.3, label=lab)
    ax.set_xlabel("time (ms)"); ax.set_ylabel("LM tibia pitch (deg)")
    ax.legend(fontsize=8, frameon=False, ncol=3)
    fig.savefig(out / "example_trace.png", dpi=160)
    print(json.dumps(res, indent=1)[:1500])


if __name__ == "__main__":
    main()
