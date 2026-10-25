"""Dataset and models for leg-state imputation on the six-leg graph."""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

LEGS = ["lf", "lm", "lh", "rf", "rm", "rh"]
N_LEGS, N_DOF = 6, 7

# Anatomical leg graph: ipsilateral neighbours + contralateral pairs.
ANATOMICAL_EDGES = [
    ("lf", "lm"), ("lm", "lh"), ("rf", "rm"), ("rm", "rh"),  # ipsilateral
    ("lf", "rf"), ("lm", "rm"), ("lh", "rh"),                # contralateral
]


def adjacency(kind: str) -> torch.Tensor:
    """Adjacency with self-loops. kind: 'anatomical' | 'full' | 'none'."""
    if kind == "full":
        return torch.ones(N_LEGS, N_LEGS)
    a = torch.eye(N_LEGS)
    if kind == "anatomical":
        for u, v in ANATOMICAL_EDGES:
            i, j = LEGS.index(u), LEGS.index(v)
            a[i, j] = a[j, i] = 1.0
    return a


# ---------------------------------------------------------------- data
def load_episodes(data_dir: str, gaits=None, seeds=None) -> list[dict]:
    eps = []
    for f in sorted(glob.glob(str(Path(data_dir) / "*.npz"))):
        d = np.load(f)
        gait, seed = str(d["gait"]), int(Path(f).stem.split("seed")[-1])
        if (gaits and gait not in gaits) or (seeds is not None and seed not in seeds):
            continue
        ang = d["angles"].reshape(-1, N_LEGS, N_DOF)          # (T, 6, 7)
        stance = (d["force"] > 1e-3).astype(np.float32)       # (T, 6)
        eps.append({"gait": gait, "seed": seed, "angles": ang, "stance": stance})
    return eps


class Normalizer:
    def __init__(self, eps):
        a = np.concatenate([e["angles"] for e in eps])        # (N, 6, 7)
        self.mu, self.sd = a.mean(0), a.std(0) + 1e-6

    def __call__(self, a):
        return (a - self.mu) / self.sd


def make_samples(eps, norm: Normalizer, window: int, delay: int):
    """Build (inputs, target) pairs for leg imputation.

    For a target leg k at time t:
      - the other legs contribute their window [t-window+1, t] (present),
      - leg k contributes only its *delayed* window [t-delay-window+1, t-delay],
      - the target is leg k's 7 joint angles at time t.
    Inputs per leg: window x (7 angles + stance) flattened.
    """
    X, M, Y, G = [], [], [], []
    for e in eps:
        a = norm(e["angles"]).astype(np.float32)
        s = e["stance"][..., None]
        feat = np.concatenate([a, s], axis=-1)                # (T, 6, 8)
        T = len(feat)
        for t in range(window + delay, T):
            present = feat[t - window + 1 : t + 1]            # (W, 6, 8)
            delayed = feat[t - delay - window + 1 : t - delay + 1]
            for k in range(N_LEGS):
                x = present.copy()
                x[:, k] = delayed[:, k]
                X.append(x.transpose(1, 0, 2).reshape(N_LEGS, -1))
                m = np.zeros(N_LEGS, np.float32); m[k] = 1
                M.append(m); Y.append(a[t, k]); G.append(e["gait"])
    return (torch.tensor(np.stack(X)), torch.tensor(np.stack(M)),
            torch.tensor(np.stack(Y)), np.array(G))


# ---------------------------------------------------------------- models
class GraphAttentionLayer(nn.Module):
    """Dense single-head attention restricted to a fixed adjacency."""

    def __init__(self, dim: int):
        super().__init__()
        self.q, self.k, self.v = (nn.Linear(dim, dim) for _ in range(3))
        self.out = nn.Sequential(nn.Linear(2 * dim, dim), nn.GELU())
        self.scale = dim ** -0.5
        self.last_attn = None

    def forward(self, h, adj):
        logits = (self.q(h) @ self.k(h).transpose(1, 2)) * self.scale
        logits = logits.masked_fill(adj[None] == 0, float("-inf"))
        attn = logits.softmax(-1)                             # (B, 6, 6): row i attends to j
        self.last_attn = attn.detach()
        msg = attn @ self.v(h)
        return h + self.out(torch.cat([h, msg], -1))


class LegGraphNet(nn.Module):
    """Message passing over the leg graph; reads out the masked (target) leg."""

    def __init__(self, in_dim: int, adj: torch.Tensor, dim: int = 96, layers: int = 2):
        super().__init__()
        self.register_buffer("adj", adj)
        self.leg_emb = nn.Parameter(torch.randn(N_LEGS, dim) * 0.1)
        self.enc = nn.Sequential(nn.Linear(in_dim + 1, dim), nn.GELU(), nn.Linear(dim, dim))
        self.layers = nn.ModuleList(GraphAttentionLayer(dim) for _ in range(layers))
        self.head = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, N_DOF))

    def forward(self, x, mask):
        h = self.enc(torch.cat([x, mask[..., None]], -1)) + self.leg_emb
        for layer in self.layers:
            h = layer(h, self.adj)
        target = (h * mask[..., None]).sum(1)
        return self.head(target)

    def attention(self):
        return [l.last_attn for l in self.layers]
