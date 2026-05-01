import torch
import numpy as np

class PPOBuffer:
    def __init__(self, size, state_shape, gamma=0.99, lam=0.95):
        # We store everything as torch tensors for speed
        self.states = torch.zeros((size, *state_shape), dtype=torch.float32)
        self.actions = torch.zeros(size, dtype=torch.float32)
        self.rewards = torch.zeros(size, dtype=torch.float32)
        self.values = torch.zeros(size, dtype=torch.float32)
        self.log_probs = torch.zeros(size, dtype=torch.float32)
        
        # Advantage and Return will be computed after the rollout
        self.advantages = torch.zeros(size, dtype=torch.float32)
        self.returns = torch.zeros(size, dtype=torch.float32)
        
        self.gamma, self.lam = gamma, lam
        self.ptr, self.max_size = 0, size

    def store(self, state, action, reward, value, log_prob):
        """Add one step of experience to the buffer."""
        if self.ptr < self.max_size:
            self.states[self.ptr] = state
            self.actions[self.ptr] = action
            self.rewards[self.ptr] = reward
            self.values[self.ptr] = value
            self.log_probs[self.ptr] = log_prob
            self.ptr += 1

    def finish_path(self, last_val=0):
        """Calculate GAE advantages and discounted returns."""
        path_slice = slice(0, self.ptr)
        # Add the last predicted value to bootstrap
        rewards = torch.cat([self.rewards[path_slice], torch.tensor([last_val])])
        values = torch.cat([self.values[path_slice], torch.tensor([last_val])])
        
        # The TD-error (delta)
        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]
        
        # Compute GAE-lambda advantage
        adv = torch.zeros_like(self.rewards[path_slice])
        last_gae = 0
        for t in reversed(range(self.ptr)):
            adv[t] = last_gae = deltas[t] + self.gamma * self.lam * last_gae
        
        self.advantages[path_slice] = adv
        self.returns[path_slice] = adv + self.values[path_slice]
        
    def get(self):
        """Return the data for training and reset pointer."""
        self.ptr = 0
        # Normalize advantages (standard PPO trick for stability)
        adv_mean, adv_std = self.advantages.mean(), self.advantages.std()
        self.advantages = (self.advantages - adv_mean) / (adv_std + 1e-8)
        
        return dict(states=self.states, actions=self.actions, 
                    returns=self.returns, log_probs=self.log_probs, 
                    advantages=self.advantages)
    
    def clear(self):
        """Reset the buffer."""
        self.ptr = 0