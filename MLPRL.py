"""
Flat MLP actor-critic for Tangram — the "point-wise encoding" strawman.

Each piece's pose (centroid x, y, shape id, orientation id, target offset x, y)
is concatenated across all pieces into one fixed vector and pushed through an
ordinary MLP. There is no per-piece structure and no permutation invariance:
a different piece ordering is a different input.
"""
import torch.nn as nn


class MLPActorCritic(nn.Module):
    def __init__(self, input_dim, num_actions, hidden_dim=128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_actions),
        )

        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        # x: [B, input_dim]
        latent = self.trunk(x)
        return self.actor(latent), self.critic(latent)
