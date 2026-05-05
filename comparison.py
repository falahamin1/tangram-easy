"""
Easy Tangram — Three-Way Representation Comparison
====================================================
Trains all three polytope-based representations and compares them:

  H-Rep  : DeepSet on half-space parameters         [4, 4, 3] → [4, 12]
  V-Rep  : DeepSet on vertex coordinates            [4, 4, 2] → [4,  8]
  GNN    : Graph Neural Network on constraint graph [4, 4, 3] + [4, 4, 4]

Usage
-----
    # Full training (1000 rollouts each, ~hours on CPU):
    python comparison.py

    # Quick smoke-test (5 rollouts each):
    python comparison.py --quick

    # Ultra-quick (2 rollouts, just to check imports):
    python comparison.py --ultraquick
"""

import sys
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from easy_env import EasyTangramGym
from DeepSetRL import DeepSetActorCritic
from GraphNNRL import GNNActorCritic
from hrep     import train_h_rep
from vrep     import train_v_rep
from graphrep import train_graph_rep


# ── Greedy evaluation ─────────────────────────────────────────────────────────

def evaluate_model(model, rep_key: str, num_steps: int = 200, render_prefix: str = ""):
    """Run one greedy episode and return total reward."""
    env         = EasyTangramGym()
    obs         = env.reset()
    total_reward = 0.0
    model.eval()

    for step in range(1, num_steps + 1):
        mask = env.get_action_mask()
        mask_ts = torch.tensor(mask, dtype=torch.bool)

        if rep_key == "gnn":
            h_rep = torch.tensor(obs["h_rep"], dtype=torch.float32).unsqueeze(0)
            adj   = torch.tensor(obs["adj"],   dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logits, _ = model(h_rep, adj)
        elif rep_key == "h_rep":
            raw   = torch.tensor(obs["h_rep"], dtype=torch.float32).view(4, 12).unsqueeze(0)
            with torch.no_grad():
                logits, _ = model(raw)
        else:   # v_rep
            raw   = torch.tensor(obs["v_rep"], dtype=torch.float32).view(4, 8).unsqueeze(0)
            with torch.no_grad():
                logits, _ = model(raw)

        logits[0][~mask_ts] = -1e10
        action = torch.argmax(logits, dim=-1).item()

        obs, reward, done, info = env.step(action)
        total_reward += reward

        if render_prefix and (step % 50 == 0 or done):
            label = "FINAL" if done else f"step{step}"
            env.inner.render(f"{render_prefix}_{label}")

        if done:
            print(f"  Solved in {step} steps!")
            break

    return total_reward


# ── Comparison plot ───────────────────────────────────────────────────────────

def plot_results(reward_histories: dict, eval_scores: dict):
    window = 20
    palette = {"H-Rep": "steelblue", "V-Rep": "darkorange", "GNN": "seagreen"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for label, rewards in reward_histories.items():
        color = palette.get(label, "gray")
        ax1.plot(rewards, alpha=0.2, color=color)
        if len(rewards) >= window:
            ma = np.convolve(rewards, np.ones(window) / window, mode="valid")
            ax1.plot(range(window - 1, len(rewards)), ma,
                     color=color, linewidth=2, label=label)

    ax1.axhline(0, color="gray", linewidth=0.6, linestyle="--")
    ax1.set_xlabel("Game episode")
    ax1.set_ylabel("Total reward")
    ax1.set_title("Easy Tangram — Training Curves")
    ax1.legend()

    labels_ = list(eval_scores.keys())
    values_ = [eval_scores[l] for l in labels_]
    colors_ = [palette.get(l, "gray") for l in labels_]
    bars    = ax2.bar(labels_, values_, color=colors_, width=0.4)
    ax2.set_ylabel("Greedy eval reward (200 steps)")
    ax2.set_title("Best-Model Evaluation")
    for bar, val in zip(bars, values_):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                 f"{val:.2f}", ha="center", va="bottom")

    fig.suptitle("Easy Tangram  ·  H-Rep vs V-Rep vs GNN", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig("easy_tangram_comparison.png", dpi=150)
    plt.close(fig)
    print("\nComparison plot saved → easy_tangram_comparison.png")


def print_summary(reward_histories: dict, eval_scores: dict):
    print("\n" + "═" * 60)
    print(f"  {'Representation':<12}  {'Avg100 (train)':>16}  {'Best':>8}  {'Eval':>8}")
    print("─" * 60)
    for label, rewards in reward_histories.items():
        avg  = float(np.mean(rewards[-100:])) if rewards else 0.0
        best = float(max(rewards))            if rewards else 0.0
        ev   = eval_scores.get(label, 0.0)
        print(f"  {label:<12}  {avg:>16.2f}  {best:>8.2f}  {ev:>8.2f}")
    print("═" * 60)


# ── Entry points ──────────────────────────────────────────────────────────────

def run_comparison(episodes: int = 5000):
    print(f"\n{'─'*55}")
    print(f"  Easy Tangram  —  {episodes} rollouts per representation")
    print(f"{'─'*55}")

    print("\n[1/3] Training H-Rep (DeepSet on half-space params)...")
    _, best_h, hist_h = train_h_rep(episodes=episodes)

    print("\n[2/3] Training V-Rep (DeepSet on vertices)...")
    _, best_v, hist_v = train_v_rep(episodes=episodes)

    print("\n[3/3] Training GNN  (constraint-graph neural network)...")
    _, best_g, hist_g = train_graph_rep(episodes=episodes)

    reward_histories = {"H-Rep": hist_h, "V-Rep": hist_v, "GNN": hist_g}

    print("\n── Greedy evaluation of best models ──")
    eval_scores = {
        "H-Rep": evaluate_model(best_h, "h_rep", render_prefix="eval_hrep"),
        "V-Rep": evaluate_model(best_v, "v_rep", render_prefix="eval_vrep"),
        "GNN"  : evaluate_model(best_g, "gnn",   render_prefix="eval_gnn"),
    }

    plot_results(reward_histories, eval_scores)
    print_summary(reward_histories, eval_scores)
    return reward_histories, eval_scores


def run_quick():
    """100 rollouts per representation — a few minutes on CPU."""
    return run_comparison(episodes=100)


def run_ultraquick():
    """5 rollouts — just enough to confirm all imports and shapes work."""
    return run_comparison(episodes=5)


if __name__ == "__main__":
    if "--ultraquick" in sys.argv:
        run_ultraquick()
    elif "--quick" in sys.argv:
        run_quick()
    else:
        run_comparison(episodes=10000)
