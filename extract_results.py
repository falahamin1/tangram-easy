"""
Read-only extractor: walks existing tangram-easy artifacts on disk
(policies/*.pth) and writes results_extracted.csv in the common cross-
benchmark flat schema. Does not train.

Two metrics-dict formats can be present:
  - New format (seeds trained after train_single.py's evaluate() was
    upgraded to dual-mode + entropy): metrics has solve_rate_greedy,
    solve_rate_stochastic, final_entropy directly -- read straight through,
    no reconstruction needed.
  - Old format (seeds 0-2, trained before that upgrade): metrics only has a
    single greedy solve_rate, no entropy, no step counts. For these, this
    script reconstructs steps/entropy/stochastic solve rate by loading the
    saved best-model weights and running frozen-weight rollouts (inference
    only, no training) in both eval modes -- this is what answers Q3 (are
    the GNN seeds that landed at 0% solve decisive-but-wrong or never
    decisive?).

One row per (method, seed, eval_mode) -- greedy and stochastic tagged
separately via the eval_mode column, per the single-fixed-puzzle-per-run
structure of this benchmark (no puzzle_idx dimension exists).
"""
import csv
import glob
import os
import re
import sys

# The cluster's saved checkpoints/policies were pickled under numpy>=2.0,
# which moved internals to numpy._core; this local env has numpy<2.0. This
# shim only affects local deserialization of already-trained artifacts, not
# any experiment logic.
import numpy.core as _np_core
import numpy.core.multiarray as _np_multiarray
sys.modules.setdefault('numpy._core', _np_core)
sys.modules.setdefault('numpy._core.multiarray', _np_multiarray)

import numpy as np
import torch
from torch.distributions import Categorical

from easy_env import EasyTangramGym, FLAT_POSE_DIM, GRID_CHANNELS
from DeepSetRL import DeepSetActorCritic
from GraphNNRL import GNNActorCritic
from MLPRL import MLPActorCritic
from CNNRL import CNNActorCritic

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
POLICY_DIR = os.path.join(REPO_DIR, 'policies')
METHODS = ['hrep', 'vrep', 'gnn', 'mlp', 'cnn']
NUM_PIECES, H_DIM, V_DIM, NUM_ACTIONS = 4, 12, 8, 16
N_ROLLOUT_EPISODES = 20

DONE_RE = re.compile(r"\[train_single\] Policy saved.*?(hrep|vrep|gnn|mlp|cnn)_seed(\d+)\.pth")


def build_completion_log_index():
    completed = set()
    for fname in glob.glob(os.path.join(REPO_DIR, '*.out')) + glob.glob(os.path.join(REPO_DIR, 'logs', '*.out')):
        try:
            with open(fname, errors='ignore') as f:
                text = f.read()
        except OSError:
            continue
        for m in DONE_RE.finditer(text):
            method, seed = m.groups()
            completed.add((method, int(seed)))
    return completed


def build_model(method):
    if method == 'hrep':
        return DeepSetActorCritic(input_dim=H_DIM, num_pieces=NUM_PIECES, num_actions=NUM_ACTIONS)
    if method == 'vrep':
        return DeepSetActorCritic(input_dim=V_DIM, num_pieces=NUM_PIECES, num_actions=NUM_ACTIONS)
    if method == 'gnn':
        return GNNActorCritic(node_dim=3, hidden_dim=128, num_actions=NUM_ACTIONS)
    if method == 'mlp':
        return MLPActorCritic(input_dim=FLAT_POSE_DIM, num_actions=NUM_ACTIONS)
    return CNNActorCritic(in_channels=GRID_CHANNELS, num_actions=NUM_ACTIONS)


def get_action_and_entropy(model, method, obs, mask, mode='greedy', rng=None):
    with torch.no_grad():
        if method == 'hrep':
            s = torch.tensor(obs['h_rep'], dtype=torch.float32).view(1, NUM_PIECES, H_DIM)
            logits, _ = model(s)
        elif method == 'vrep':
            s = torch.tensor(obs['v_rep'], dtype=torch.float32).view(1, NUM_PIECES, V_DIM)
            logits, _ = model(s)
        elif method == 'gnn':
            h = torch.tensor(obs['h_rep'], dtype=torch.float32).unsqueeze(0)
            adj = torch.tensor(obs['adj'], dtype=torch.float32).unsqueeze(0)
            logits, _ = model(h, adj)
        elif method == 'mlp':
            s = torch.tensor(obs['flat_pose'], dtype=torch.float32).unsqueeze(0)
            logits, _ = model(s)
        else:
            s = torch.tensor(obs['grid_image'], dtype=torch.float32).unsqueeze(0)
            logits, _ = model(s)
        logits[0][~mask] = -1e10
        dist = Categorical(logits=logits)
        entropy = dist.entropy().item()
        if mode == 'greedy':
            action = torch.argmax(logits, dim=-1).item()
        else:
            probs = torch.softmax(logits, dim=-1).squeeze(0).numpy()
            probs = probs / probs.sum()
            action = int(rng.choice(len(probs), p=probs))
        return action, entropy


def rollout(model, method, mode, n_episodes=N_ROLLOUT_EPISODES, seed=0):
    """Rollouts on the fixed puzzle in the given mode: solve rate, mean step
    count on solved episodes (or None if none solved), mean policy entropy
    over every visited step."""
    model.eval()
    rng = np.random.default_rng(seed)
    solved_steps, entropies, solves = [], [], 0
    for _ in range(n_episodes):
        env = EasyTangramGym()
        obs = env.reset()
        for t in range(env.max_steps):
            mask = torch.tensor(env.get_action_mask(), dtype=torch.bool)
            action, ent = get_action_and_entropy(model, method, obs, mask, mode=mode, rng=rng)
            entropies.append(ent)
            obs, _r, done, info = env.step(action)
            if done:
                if info.get('completion', 0) >= 1.0:
                    solves += 1
                    solved_steps.append(t + 1)
                break
    mean_steps = float(np.mean(solved_steps)) if solved_steps else None
    mean_entropy = float(np.mean(entropies))
    return solves / n_episodes, mean_steps, mean_entropy


def main():
    completed_log_index = build_completion_log_index()

    rows = []
    for path in sorted(glob.glob(os.path.join(POLICY_DIR, '*.pth'))):
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        method = ckpt['method']
        seed = ckpt['seed']
        metrics = ckpt['metrics']
        provenance = 'trusted' if (method, seed) in completed_log_index else 'unverified'

        if 'solve_rate_stochastic' in metrics:
            per_mode = {
                'greedy': (metrics['solve_rate_greedy'], None),
                'stochastic': (metrics['solve_rate_stochastic'], None),
            }
            final_entropy = metrics['final_entropy']
        else:
            model = build_model(method)
            model.load_state_dict(ckpt['model_state'])
            g_rate, g_steps, g_entropy = rollout(model, method, 'greedy', seed=seed)
            s_rate, s_steps, _s_entropy = rollout(model, method, 'stochastic', seed=seed)
            per_mode = {'greedy': (g_rate, g_steps), 'stochastic': (s_rate, s_steps)}
            final_entropy = g_entropy

        for mode, (solve_rate, mean_steps) in per_mode.items():
            rows.append({
                'benchmark': 'tangram',
                'tier': 'easy',
                'method': method,
                'seed': seed,
                'instance_id': 'easy_fixed',
                'eval_mode': mode,
                'solved': 1 if solve_rate >= 0.5 else 0,
                'steps': f'{mean_steps:.1f}' if mean_steps is not None else '',
                'optimal_steps': '',   # no ground-truth optimal move count exists for this puzzle
                'rollouts_to_solve': '',  # see Step 4 validation, separate from this extractor
                'final_entropy': f'{final_entropy:.4f}',
                'provenance': provenance,
            })

    out_path = os.path.join(REPO_DIR, 'results_extracted.csv')
    fieldnames = ['benchmark', 'tier', 'method', 'seed', 'instance_id', 'eval_mode', 'solved',
                  'steps', 'optimal_steps', 'rollouts_to_solve', 'final_entropy', 'provenance']
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (r['method'], r['seed'], r['eval_mode'])))

    trusted = sum(1 for r in rows if r['provenance'] == 'trusted')
    unverified = sum(1 for r in rows if r['provenance'] == 'unverified')
    print(f"Wrote {len(rows)} rows to {out_path}")
    print(f"provenance: trusted={trusted}  unverified={unverified}")


if __name__ == '__main__':
    main()
