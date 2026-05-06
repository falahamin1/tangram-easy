"""
Plot training curves from saved checkpoints.

Handles checkpoints saved on numpy-2.x HPC (numpy._core) loaded on numpy-1.x locally.

Usage:
    python plot_training.py                  # saves easy_tangram_training.png
    python plot_training.py --out my_fig.png
"""

import argparse
import io
import sys
import types

# ── numpy 1.x / 2.x compatibility ────────────────────────────────────────────
import numpy
import numpy.core

_compat = types.ModuleType("numpy._core")
_compat.__dict__.update(numpy.core.__dict__)
sys.modules["numpy._core"] = _compat
for _sub in ["multiarray", "numeric", "fromnumeric", "function_base",
             "shape_base", "umath", "_multiarray_umath"]:
    _src = sys.modules.get(f"numpy.core.{_sub}")
    if _src:
        sys.modules[f"numpy._core.{_sub}"] = _src
# ─────────────────────────────────────────────────────────────────────────────

import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
PALETTE = {"H-Rep": "steelblue", "V-Rep": "darkorange", "GNN": "seagreen"}
METHODS = [("hrep", "H-Rep"), ("vrep", "V-Rep"), ("gnn", "GNN")]


def load_history(method):
    path = os.path.join(CHECKPOINT_DIR, method, f"checkpoint_ep0019999.pth")
    if not os.path.exists(path):
        # fall back to any checkpoint
        import glob
        ckpts = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, method, "checkpoint_ep*.pth")))
        if not ckpts:
            print(f"  [SKIP] no checkpoint found for {method}")
            return None, None
        path = ckpts[-1]
    data = torch.load(path, map_location="cpu", weights_only=False)
    return np.array(data["reward_history"]), data["best_moving_avg"]


def moving_average(arr, window=200):
    if len(arr) < window:
        return np.arange(len(arr)), arr
    ma = np.convolve(arr, np.ones(window) / window, mode="valid")
    return np.arange(window - 1, len(arr)), ma


def plot(out_path):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    ax_curve, ax_bar = axes

    solve_scores = {}

    for key, label in METHODS:
        hist, best_avg = load_history(key)
        if hist is None:
            continue

        color  = PALETTE[label]
        solved = hist >= 13.5

        x_ma, ma = moving_average(hist, window=200)
        ax_curve.plot(hist, alpha=0.08, color=color, linewidth=0.4)
        ax_curve.plot(x_ma, ma, color=color, linewidth=2,
                      label=f"{label}  (best={best_avg:.2f})")

        # solve rate in rolling 1 000-ep window
        sr = np.convolve(solved.astype(float), np.ones(1000) / 1000, mode="valid")
        ax_bar_x = np.arange(999, len(hist))

        n      = len(hist)
        sr_end = solved[-1000:].mean() * 100
        solve_scores[label] = (best_avg, sr_end)

        print(f"{label:6s}  n={n:>7,}  best_avg={best_avg:.2f}  "
              f"solve_rate_last1000={sr_end:.1f}%")

    ax_curve.axhline(0, color="gray", linewidth=0.6, linestyle="--")
    ax_curve.set_xlabel("Episode")
    ax_curve.set_ylabel("Episode reward")
    ax_curve.set_title("Easy Tangram — Training Curves (200-ep moving avg)")
    ax_curve.legend(fontsize=9)

    labels_ = [l for _, l in METHODS if l in solve_scores]
    best_   = [solve_scores[l][0] for l in labels_]
    sr_     = [solve_scores[l][1] for l in labels_]
    colors_ = [PALETTE[l] for l in labels_]
    x_pos   = np.arange(len(labels_))
    width   = 0.35

    b1 = ax_bar.bar(x_pos - width / 2, best_, width, color=colors_, alpha=0.9,
                    label="Best moving avg reward")
    b2 = ax_bar.bar(x_pos + width / 2, sr_,   width, color=colors_, alpha=0.45,
                    label="Solve rate last 1 000 eps (%)")
    for bar, val in zip(b1, best_):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8)
    for bar, val in zip(b2, sr_):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                    f"{val:.0f}%", ha="center", va="bottom", fontsize=8)

    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(labels_)
    ax_bar.set_title("Best Reward vs Solve Rate (last 1 000 episodes)")
    ax_bar.legend(fontsize=8)

    fig.suptitle("Easy Tangram  ·  H-Rep vs V-Rep vs GNN  (20 000 rollouts)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nPlot saved → {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="easy_tangram_training.png")
    args = ap.parse_args()
    plot(args.out)
