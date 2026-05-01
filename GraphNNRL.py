"""
Graph Neural Network Actor-Critic for the Easy Tangram (4 pieces).

Identical architecture to the hard-tangram GNN, adapted for:
  - 4 pieces  × 4 constraints = 16 nodes  (vs 6 × 5 = 30 in the hard version)
  - 16 actions                             (vs 24)
  - Block-diagonal adjacency: 4 blocks of 4×4 on a 16×16 matrix
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.Module):
    """Standard Graph Convolutional Layer (message-passing + linear projection)."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.projection = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        # x   : [Batch, Nodes, Feats]
        # adj : [Batch, Nodes, Nodes]
        support = torch.bmm(adj, x)           # sum neighbor features
        return F.relu(self.projection(support))


class GNNActorCritic(nn.Module):
    """
    GNN Actor-Critic for the easy 4-square tangram.

    Inputs to forward():
        h_rep : [Batch, 4, 4, 3]  — 4 pieces, 4 constraints, 3 params (a1,a2,b)
        adj   : [Batch, 4, 4, 4]  — per-piece constraint adjacency matrices

    The 4 per-piece adjacency matrices are embedded in a block-diagonal
    16×16 adjacency (with self-loops added), so all 16 constraint-nodes are
    processed jointly in one graph.
    """

    def __init__(self, node_dim: int = 3, hidden_dim: int = 128, num_actions: int = 16):
        super().__init__()

        # GNN encoder: two graph-convolutional layers
        self.gcn1 = GCNLayer(node_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, hidden_dim)

        # Shared refinement after global pooling
        self.rho = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Policy head
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_actions),
        )

        # Value head
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, h_rep, adj):
        batch_size = h_rep.shape[0]

        # Flatten 4 pieces × 4 constraints into 16 nodes: [Batch, 16, 3]
        x = h_rep.view(batch_size, -1, 3)

        # Build block-diagonal 16×16 adjacency from 4 per-piece 4×4 matrices
        big_adj = torch.zeros(batch_size, 16, 16, device=h_rep.device)
        for i in range(4):
            big_adj[:, i * 4:(i + 1) * 4, i * 4:(i + 1) * 4] = adj[:, i, :, :]

        # Add self-loops so each node aggregates its own features
        big_adj = big_adj + torch.eye(16, device=h_rep.device).unsqueeze(0)

        # GNN layers
        h = self.gcn1(x, big_adj)    # [Batch, 16, hidden_dim]
        h = self.gcn2(h, big_adj)    # [Batch, 16, hidden_dim]

        # Global max-pooling over all 16 nodes
        global_pool = torch.max(h, dim=1)[0]   # [Batch, hidden_dim]
        latent      = self.rho(global_pool)

        return self.actor(latent), self.critic(latent)
