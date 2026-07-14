"""
CNN actor-critic for Tangram — the "encoder every reviewer will imagine".

The board is encoded as a multi-channel occupancy image (per-piece occupancy,
per-piece target silhouette, locked mask) and pushed through a small CNN.
This gives the encoder local geometric structure (unlike the flat MLP) but
still no permutation-invariance over pieces — it reasons over grid cells,
not over a set of pieces.
"""
import torch.nn as nn

from easy_env import GRID


class CNNActorCritic(nn.Module):
    def __init__(self, in_channels, num_actions, hidden_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.trunk = nn.Sequential(
            nn.Linear(64 * GRID * GRID, hidden_dim),
            nn.LayerNorm(hidden_dim),
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
        # x: [B, in_channels, GRID, GRID]
        feats  = self.conv(x).flatten(start_dim=1)
        latent = self.trunk(feats)
        return self.actor(latent), self.critic(latent)
