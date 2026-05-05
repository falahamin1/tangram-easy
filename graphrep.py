"""
Easy Tangram — Graph Neural Network PPO Training
================================================
Trains a GNN actor-critic using the constraint graph of each piece.

Dual state: h_rep [4, 4, 3] (node features) + adj [4, 4, 4] (edges).
These are assembled into a 16-node block-diagonal graph inside GNNActorCritic.

Run:
    python graphrep.py
"""

import os
import glob
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from copy import deepcopy

from easy_env import EasyTangramGym
from GraphNNRL import GNNActorCritic
from PPOBuffer import PPOBuffer

_KEEP = 3


def _save_checkpoint(ckpt_dir, ep, model, opt,
                     reward_history, best_moving_avg, best_weights):
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f'checkpoint_ep{ep:07d}.pth')
    tmp  = path + '.tmp'
    torch.save({
        'episode':              ep,
        'model_state_dict':     model.state_dict(),
        'optimizer_state_dict': opt.state_dict(),
        'reward_history':       reward_history,
        'best_moving_avg':      best_moving_avg,
        'best_weights':         best_weights,
    }, tmp)
    os.replace(tmp, path)
    torch.save(best_weights, os.path.join(ckpt_dir, 'gnn_best.pth'))
    for f in sorted(glob.glob(os.path.join(ckpt_dir, 'checkpoint_ep*.pth')))[:-_KEEP]:
        try:
            os.remove(f)
        except OSError:
            pass
    print(f"[GNN] checkpoint saved → ep={ep}")


def _load_latest_checkpoint(ckpt_dir, model, opt):
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, 'checkpoint_ep*.pth')))
    if not ckpts:
        return 0, [], -float('inf'), None
    latest = ckpts[-1]
    print(f"[GNN] Resuming from {latest}")
    data = torch.load(latest, map_location='cpu')
    model.load_state_dict(data['model_state_dict'])
    opt.load_state_dict(data['optimizer_state_dict'])
    return (data['episode'] + 1, data['reward_history'],
            data['best_moving_avg'], data['best_weights'])


def train_graph_rep(episodes: int = 1000,
                    checkpoint_dir: str = 'checkpoints/gnn',
                    checkpoint_interval: int = 500):
    HP = {
        "lr"               : 1e-4,
        "clip_eps"         : 0.2,
        "ppo_epochs"       : 5,
        "steps_per_rollout": 1024,
        "batch_size"       : 64,
        "gamma"            : 0.99,
        "entropy_coef"     : 0.05,
        "critic_coef"      : 0.5,
        "max_grad_norm"    : 0.5,
        "moving_avg_window": 20,
    }

    env   = EasyTangramGym()
    model = GNNActorCritic(node_dim=3, hidden_dim=128, num_actions=16)
    opt   = optim.Adam(model.parameters(), lr=HP["lr"])

    h_buffer   = PPOBuffer(size=HP["steps_per_rollout"], state_shape=(4, 4, 3))
    adj_buffer = PPOBuffer(size=HP["steps_per_rollout"], state_shape=(4, 4, 4))

    start_ep, reward_history, best_moving_avg, loaded_best = \
        _load_latest_checkpoint(checkpoint_dir, model, opt)
    best_weights = loaded_best if loaded_best is not None else deepcopy(model.state_dict())

    print(f"[GNN] episodes={episodes}  start_ep={start_ep}  ckpt_dir={checkpoint_dir}")

    for ep in range(start_ep, episodes):
        obs       = env.reset()
        ep_reward = 0

        for t in range(HP["steps_per_rollout"]):
            h_rep = torch.tensor(obs["h_rep"], dtype=torch.float32).unsqueeze(0)
            adj   = torch.tensor(obs["adj"],   dtype=torch.float32).unsqueeze(0)

            mask_ts = torch.tensor(env.get_action_mask(), dtype=torch.bool)

            with torch.no_grad():
                logits, value = model(h_rep, adj)
                logits[0][~mask_ts] = -1e10
                dist   = Categorical(logits=logits)
                action = dist.sample()
                logp   = dist.log_prob(action)

            next_obs, reward, done, info = env.step(action.item())

            h_buffer.store(h_rep.squeeze(0),  action, reward, value.item(), logp.item())
            adj_buffer.store(adj.squeeze(0),  action, 0, 0, 0)

            obs       = next_obs
            ep_reward += reward

            if done:
                h_buffer.finish_path(last_val=0)
                reward_history.append(ep_reward)
                obs       = env.reset()
                ep_reward = 0
            elif t == HP["steps_per_rollout"] - 1:
                h_next   = torch.tensor(obs["h_rep"], dtype=torch.float32).unsqueeze(0)
                adj_next = torch.tensor(obs["adj"],   dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    _, last_val = model(h_next, adj_next)
                h_buffer.finish_path(last_val.item())

        if len(reward_history) >= HP["moving_avg_window"]:
            avg = np.mean(reward_history[-HP["moving_avg_window"]:])
            if avg > best_moving_avg:
                best_moving_avg = avg
                best_weights    = deepcopy(model.state_dict())
                print(f"*** NEW BEST GNN    (avg={best_moving_avg:.2f}) at rollout {ep} ***")

        data_h   = h_buffer.get()
        data_adj = adj_buffer.get()
        indices  = np.arange(HP["steps_per_rollout"])
        for _ in range(HP["ppo_epochs"]):
            np.random.shuffle(indices)
            for start in range(0, HP["steps_per_rollout"], HP["batch_size"]):
                mb       = indices[start : start + HP["batch_size"]]
                mb_h     = data_h["states"][mb]
                mb_adj   = data_adj["states"][mb]
                mb_a     = data_h["actions"][mb]
                mb_adv   = data_h["advantages"][mb]
                mb_ret   = data_h["returns"][mb]
                mb_oldlp = data_h["log_probs"][mb]

                logits, values = model(mb_h, mb_adj)
                dist    = Categorical(logits=logits)
                new_lp  = dist.log_prob(mb_a)
                entropy = dist.entropy().mean()

                ratio  = torch.exp(new_lp - mb_oldlp)
                surr1  = ratio * mb_adv
                surr2  = torch.clamp(ratio, 1 - HP["clip_eps"], 1 + HP["clip_eps"]) * mb_adv
                a_loss = -torch.min(surr1, surr2).mean()
                c_loss = F.mse_loss(values.squeeze(-1), mb_ret)
                loss   = a_loss + HP["critic_coef"] * c_loss - HP["entropy_coef"] * entropy

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), HP["max_grad_norm"])
                opt.step()

        h_buffer.clear()
        adj_buffer.clear()

        if ep % 100 == 0:
            avg = np.mean(reward_history[-10:]) if reward_history else 0
            print(f"GNN    rollout {ep:5d} | recent={avg:.2f} | best={best_moving_avg:.2f}")

        if checkpoint_interval > 0 and ep > 0 and ep % checkpoint_interval == 0:
            _save_checkpoint(checkpoint_dir, ep, model, opt,
                             reward_history, best_moving_avg, best_weights)

    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'gnn_final.pth'))
    _save_checkpoint(checkpoint_dir, episodes - 1, model, opt,
                     reward_history, best_moving_avg, best_weights)
    print(f"[GNN] Done. Models saved to {checkpoint_dir}/")

    best_model = GNNActorCritic(node_dim=3, hidden_dim=128, num_actions=16)
    best_model.load_state_dict(best_weights)
    return model, best_model, reward_history


if __name__ == "__main__":
    train_graph_rep(episodes=1000)
