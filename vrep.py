"""
Easy Tangram — V-Representation PPO Training
=============================================
Trains a DeepSet actor-critic using the vertex representation (V-rep)
of each piece's PPL polytope as state.

State shape fed to the network: [batch, 4, 8]
  4 pieces × (4 vertices × 2 coords) = 8 features per piece

Run:
    python vrep.py
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
from DeepSetRL import DeepSetActorCritic
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
    torch.save(best_weights, os.path.join(ckpt_dir, 'vrep_best.pth'))
    for f in sorted(glob.glob(os.path.join(ckpt_dir, 'checkpoint_ep*.pth')))[:-_KEEP]:
        try:
            os.remove(f)
        except OSError:
            pass
    print(f"[V-rep] checkpoint saved → ep={ep}")


def _load_latest_checkpoint(ckpt_dir, model, opt):
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, 'checkpoint_ep*.pth')))
    if not ckpts:
        return 0, [], -float('inf'), None
    latest = ckpts[-1]
    print(f"[V-rep] Resuming from {latest}")
    data = torch.load(latest, map_location='cpu')
    model.load_state_dict(data['model_state_dict'])
    opt.load_state_dict(data['optimizer_state_dict'])
    return (data['episode'] + 1, data['reward_history'],
            data['best_moving_avg'], data['best_weights'])


def train_v_rep(episodes: int = 1000,
                checkpoint_dir: str = 'checkpoints/vrep',
                checkpoint_interval: int = 500):
    HP = {
        "lr"               : 3e-4,
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

    env    = EasyTangramGym()
    model  = DeepSetActorCritic(input_dim=8, num_pieces=4, num_actions=16)
    opt    = optim.Adam(model.parameters(), lr=HP["lr"])
    buffer = PPOBuffer(size=HP["steps_per_rollout"], state_shape=(4, 8))

    start_ep, reward_history, best_moving_avg, loaded_best = \
        _load_latest_checkpoint(checkpoint_dir, model, opt)
    best_weights = loaded_best if loaded_best is not None else deepcopy(model.state_dict())

    print(f"[V-rep] episodes={episodes}  start_ep={start_ep}  ckpt_dir={checkpoint_dir}")

    for ep in range(start_ep, episodes):
        obs       = env.reset()
        ep_reward = 0

        for t in range(HP["steps_per_rollout"]):
            raw      = torch.tensor(obs["v_rep"], dtype=torch.float32)
            state_f  = raw.view(4, 8)
            state_in = state_f.unsqueeze(0)

            mask_ts = torch.tensor(env.get_action_mask(), dtype=torch.bool)

            with torch.no_grad():
                logits, value = model(state_in)
                logits[0][~mask_ts] = -1e10
                dist   = Categorical(logits=logits)
                action = dist.sample()
                logp   = dist.log_prob(action)

            next_obs, reward, done, info = env.step(action.item())
            buffer.store(state_f, action, reward, value.item(), logp.item())
            obs       = next_obs
            ep_reward += reward

            if done:
                buffer.finish_path(last_val=0)
                reward_history.append(ep_reward)
                obs       = env.reset()
                ep_reward = 0
            elif t == HP["steps_per_rollout"] - 1:
                raw_next = torch.tensor(obs["v_rep"], dtype=torch.float32)
                with torch.no_grad():
                    _, last_val = model(raw_next.view(4, 8).unsqueeze(0))
                buffer.finish_path(last_val.item())

        if len(reward_history) >= HP["moving_avg_window"]:
            avg = np.mean(reward_history[-HP["moving_avg_window"]:])
            if avg > best_moving_avg:
                best_moving_avg = avg
                best_weights    = deepcopy(model.state_dict())
                print(f"*** NEW BEST V-REP  (avg={best_moving_avg:.2f}) at rollout {ep} ***")

        data    = buffer.get()
        indices = np.arange(HP["steps_per_rollout"])
        for _ in range(HP["ppo_epochs"]):
            np.random.shuffle(indices)
            for start in range(0, HP["steps_per_rollout"], HP["batch_size"]):
                mb       = indices[start : start + HP["batch_size"]]
                mb_s     = data["states"][mb]
                mb_a     = data["actions"][mb]
                mb_adv   = data["advantages"][mb]
                mb_ret   = data["returns"][mb]
                mb_oldlp = data["log_probs"][mb]

                logits, values = model(mb_s)
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

        buffer.clear()

        if ep % 100 == 0:
            avg = np.mean(reward_history[-10:]) if reward_history else 0
            print(f"V-Rep rollout {ep:5d} | recent={avg:.2f} | best={best_moving_avg:.2f}")

        if checkpoint_interval > 0 and ep > 0 and ep % checkpoint_interval == 0:
            _save_checkpoint(checkpoint_dir, ep, model, opt,
                             reward_history, best_moving_avg, best_weights)

    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'vrep_final.pth'))
    _save_checkpoint(checkpoint_dir, episodes - 1, model, opt,
                     reward_history, best_moving_avg, best_weights)
    print(f"[V-rep] Done. Models saved to {checkpoint_dir}/")

    best_model = DeepSetActorCritic(input_dim=8, num_pieces=4, num_actions=16)
    best_model.load_state_dict(best_weights)
    return model, best_model, reward_history


if __name__ == "__main__":
    train_v_rep(episodes=1000)
