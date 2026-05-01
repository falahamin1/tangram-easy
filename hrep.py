"""
Easy Tangram — H-Representation PPO Training
=============================================
Trains a DeepSet actor-critic using the half-space representation (H-rep)
of each piece's PPL polytope as state.

State shape fed to the network: [batch, 4, 12]
  4 pieces × (4 constraints × 3 params) = 12 features per piece

Run:
    python hrep.py
"""

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from copy import deepcopy

from easy_env import EasyTangramGym
from DeepSetRL import DeepSetActorCritic
from PPOBuffer import PPOBuffer


def train_h_rep(episodes: int = 1000):
    # ── 1. Hyperparameters ────────────────────────────────────────────────────
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
        "render_interval"  : 500,
        "moving_avg_window": 20,
    }

    # ── 2. Initialisation ─────────────────────────────────────────────────────
    env    = EasyTangramGym()

    # H-rep: 4 constraints × 3 params = 12 features per piece
    model  = DeepSetActorCritic(input_dim=12, num_pieces=4, num_actions=16)
    opt    = optim.Adam(model.parameters(), lr=HP["lr"])
    buffer = PPOBuffer(size=HP["steps_per_rollout"], state_shape=(4, 12))

    reward_history   = []
    best_moving_avg  = -float("inf")
    best_weights     = deepcopy(model.state_dict())

    # ── 3. Training loop ──────────────────────────────────────────────────────
    for ep in range(episodes):
        obs      = env.reset()
        ep_reward = 0

        # Rollout collection
        for t in range(HP["steps_per_rollout"]):
            # Flatten H-rep: [4, 4, 3] → [4, 12]
            raw     = torch.tensor(obs["h_rep"], dtype=torch.float32)
            state_f = raw.view(4, 12)
            state_in = state_f.unsqueeze(0)   # [1, 4, 12]

            mask    = env.get_action_mask()
            mask_ts = torch.tensor(mask, dtype=torch.bool)

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
                raw_next = torch.tensor(obs["h_rep"], dtype=torch.float32)
                _, last_val = model(raw_next.view(4, 12).unsqueeze(0))
                buffer.finish_path(last_val.item())

        # Best-model tracking
        if len(reward_history) >= HP["moving_avg_window"]:
            avg = np.mean(reward_history[-HP["moving_avg_window"]:])
            if avg > best_moving_avg:
                best_moving_avg = avg
                best_weights    = deepcopy(model.state_dict())
                print(f"*** NEW BEST H-REP  (avg={best_moving_avg:.2f}) at rollout {ep} ***")

        # PPO update
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

        # Logging & renders
        if ep % 100 == 0:
            avg = np.mean(reward_history[-10:]) if reward_history else 0
            print(f"H-Rep rollout {ep:4d} | recent={avg:.2f} | best={best_moving_avg:.2f}")

        if ep % HP["render_interval"] == 0:
            env.inner.render(f"H-Rep_Easy_Ep_{ep}")

    # ── 4. Save ───────────────────────────────────────────────────────────────
    torch.save(model.state_dict(), "model_h_rep_easy_final.pth")

    best_model = DeepSetActorCritic(input_dim=12, num_pieces=4, num_actions=16)
    best_model.load_state_dict(best_weights)
    return model, best_model, reward_history


if __name__ == "__main__":
    train_h_rep(episodes=1000)
