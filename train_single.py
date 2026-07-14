"""
Train one (method, seed) combo on the easy tangram puzzle and save the best
policy + eval metrics, for later cross-seed aggregation by
plot_tangram_results.py.

Saved to: policies/{method}_seed{seed}.pth

Seed 0 uses the long-lived checkpoints/<method>/ directory (so an
already-fully-trained hrep/vrep/gnn run is picked up and just re-evaluated,
not retrained); seed > 0 uses checkpoints/<method>/seed{seed}/.

Usage:
  python train_single.py --method hrep --seed 0 --episodes 40000
  python train_single.py --method mlp  --seed 1 --episodes 40000
"""

import os
import sys
import time
import argparse
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from easy_env import EasyTangramGym
from hrep import train_h_rep
from vrep import train_v_rep
from graphrep import train_graph_rep
from mlp import train_mlp
from cnn import train_cnn

NUM_PIECES = 4
H_DIM = 4 * 3   # 12
V_DIM = 4 * 2   # 8

POLICY_DIR = os.path.join(os.path.dirname(__file__), 'policies')
CKPT_BASE  = os.path.join(os.path.dirname(__file__), 'checkpoints')

TRAIN = {
    'hrep': train_h_rep,
    'vrep': train_v_rep,
    'gnn':  train_graph_rep,
    'mlp':  train_mlp,
    'cnn':  train_cnn,
}


def _get_action(model, method, obs, mask):
    with torch.no_grad():
        if method == 'hrep':
            s = torch.tensor(obs['h_rep'], dtype=torch.float32).view(1, NUM_PIECES, H_DIM)
            logits, _ = model(s)
        elif method == 'vrep':
            s = torch.tensor(obs['v_rep'], dtype=torch.float32).view(1, NUM_PIECES, V_DIM)
            logits, _ = model(s)
        elif method == 'gnn':
            h   = torch.tensor(obs['h_rep'], dtype=torch.float32).unsqueeze(0)
            adj = torch.tensor(obs['adj'],   dtype=torch.float32).unsqueeze(0)
            logits, _ = model(h, adj)
        elif method == 'mlp':
            s = torch.tensor(obs['flat_pose'], dtype=torch.float32).unsqueeze(0)
            logits, _ = model(s)
        else:  # cnn
            s = torch.tensor(obs['grid_image'], dtype=torch.float32).unsqueeze(0)
            logits, _ = model(s)
        logits[0][~mask] = -1e10
        return torch.argmax(logits, dim=-1).item()


def evaluate(model, method, eval_episodes=50):
    model.eval()
    rewards, solves = [], 0
    for _ in range(eval_episodes):
        env = EasyTangramGym()
        obs = env.reset()
        total = 0.0
        for _ in range(env.max_steps):
            mask = torch.tensor(env.get_action_mask(), dtype=torch.bool)
            action = _get_action(model, method, obs, mask)
            obs, r, done, info = env.step(action)
            total += r
            if done:
                if info.get('completion', 0) >= 1.0:
                    solves += 1
                break
        rewards.append(total)
    return {
        'mean_reward': float(np.mean(rewards)),
        'std_reward':  float(np.std(rewards)),
        'solve_rate':  solves / eval_episodes,
    }


def main():
    ap = argparse.ArgumentParser(description='Train one easy-tangram (method, seed) combo')
    ap.add_argument('--method', choices=list(TRAIN.keys()), required=True)
    ap.add_argument('--seed',   type=int, default=0)
    ap.add_argument('--episodes', type=int, required=True)
    ap.add_argument('--checkpoint-interval', type=int, default=500)
    ap.add_argument('--eval-episodes', type=int, default=50)
    args = ap.parse_args()

    out_path = os.path.join(POLICY_DIR, f'{args.method}_seed{args.seed}.pth')
    if os.path.exists(out_path):
        print(f"[train_single] {out_path} already exists — skipping.")
        return

    ckpt_dir = os.path.join(CKPT_BASE, args.method) if args.seed == 0 \
        else os.path.join(CKPT_BASE, args.method, f'seed{args.seed}')

    print(f"[train_single] method={args.method}  seed={args.seed}  "
          f"episodes={args.episodes}  ckpt_dir={ckpt_dir}")

    t0 = time.time()
    _, best_model = TRAIN[args.method](
        episodes=args.episodes,
        checkpoint_dir=ckpt_dir,
        checkpoint_interval=args.checkpoint_interval,
        seed=args.seed,
    )
    elapsed = time.time() - t0

    print(f"\n[train_single] Training done in {elapsed/60:.1f} min. Evaluating ...")
    metrics = evaluate(best_model, args.method, eval_episodes=args.eval_episodes)
    print(f"[train_single] solve_rate={metrics['solve_rate']*100:.1f}%  "
          f"reward={metrics['mean_reward']:.2f}±{metrics['std_reward']:.2f}")

    os.makedirs(POLICY_DIR, exist_ok=True)
    torch.save({
        'method':         args.method,
        'seed':           args.seed,
        'model_state':    best_model.state_dict(),
        'metrics':        metrics,
        'train_episodes': args.episodes,
        'train_seconds':  elapsed,
    }, out_path)
    print(f"[train_single] Policy saved → {out_path}")


if __name__ == '__main__':
    main()
