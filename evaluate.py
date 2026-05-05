"""
Evaluate saved easy-tangram policies — one greedy episode per method,
renders saved for every step.

Usage:
  python evaluate.py                   # all three methods
  python evaluate.py --method hrep     # one method only
"""

import os
import sys
import argparse
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from easy_env import EasyTangramGym
from DeepSetRL import DeepSetActorCritic
from GraphNNRL import GNNActorCritic

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), 'checkpoints')
RENDER_DIR     = os.path.join(os.path.dirname(__file__), 'eval_renders')


def load_model(method, checkpoint_dir):
    if method == 'hrep':
        model = DeepSetActorCritic(input_dim=12, num_pieces=4, num_actions=16)
        path  = os.path.join(checkpoint_dir, 'hrep', 'hrep_best.pth')
    elif method == 'vrep':
        model = DeepSetActorCritic(input_dim=8, num_pieces=4, num_actions=16)
        path  = os.path.join(checkpoint_dir, 'vrep', 'vrep_best.pth')
    else:  # gnn
        model = GNNActorCritic(node_dim=3, hidden_dim=128, num_actions=16)
        path  = os.path.join(checkpoint_dir, 'gnn', 'gnn_best.pth')

    if not os.path.exists(path):
        print(f"  [SKIP] checkpoint not found: {path}")
        return None
    model.load_state_dict(torch.load(path, map_location='cpu', weights_only=False))
    model.eval()
    print(f"  Loaded: {path}")
    return model


def get_action(model, method, obs, mask):
    with torch.no_grad():
        mask_ts = torch.tensor(mask, dtype=torch.bool)
        if method == 'hrep':
            state = torch.tensor(obs['h_rep'], dtype=torch.float32).view(1, 4, 12)
            logits, _ = model(state)
        elif method == 'vrep':
            state = torch.tensor(obs['v_rep'], dtype=torch.float32).view(1, 4, 8)
            logits, _ = model(state)
        else:  # gnn
            h   = torch.tensor(obs['h_rep'], dtype=torch.float32).unsqueeze(0)
            adj = torch.tensor(obs['adj'],   dtype=torch.float32).unsqueeze(0)
            logits, _ = model(h, adj)
        logits[0][~mask_ts] = -1e10
        return torch.argmax(logits, dim=-1).item()


def evaluate_method(method, checkpoint_dir):
    print(f"\n{'='*50}")
    print(f"  {method.upper()}")
    print(f"{'='*50}")

    model = load_model(method, checkpoint_dir)
    if model is None:
        return

    render_dir = os.path.join(RENDER_DIR, method)
    os.makedirs(render_dir, exist_ok=True)

    env = EasyTangramGym()
    obs = env.reset()
    total_reward = 0.0

    env.inner.render(os.path.join(render_dir, 'step_000.png'))
    print(f"  step 000: initial state")

    for step in range(1, env.max_steps + 1):
        mask   = env.get_action_mask()
        action = get_action(model, method, obs, mask)
        obs, reward, done, info = env.step(action)
        total_reward += reward

        env.inner.render(os.path.join(render_dir, f'step_{step:03d}.png'))
        completion = info.get('completion', 0)
        print(f"  step {step:03d}: action={action:3d}  reward={reward:+.3f}"
              f"  total={total_reward:+.3f}  pieces={completion*4:.0f}/4")

        if done:
            solved = completion >= 1.0
            print(f"\n  {'SOLVED' if solved else 'TIME LIMIT'}  "
                  f"in {step} steps  |  total reward = {total_reward:.3f}")
            break

    print(f"  Renders → {render_dir}/")


def main():
    ap = argparse.ArgumentParser(description='Evaluate saved easy-tangram policies')
    ap.add_argument('--method', choices=['hrep', 'vrep', 'gnn', 'all'], default='all')
    ap.add_argument('--checkpoint-dir',
                    default=os.path.join(os.path.dirname(__file__), 'checkpoints'))
    args = ap.parse_args()

    methods = ['hrep', 'vrep', 'gnn'] if args.method == 'all' else [args.method]
    for m in methods:
        evaluate_method(m, args.checkpoint_dir)


if __name__ == '__main__':
    main()
