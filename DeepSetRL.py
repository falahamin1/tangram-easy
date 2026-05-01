import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepSetEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(DeepSetEncoder, self).__init__()
        # phi(x): The transformation applied to each element of the set
        self.phi = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), # Better stability for RL
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

    def forward(self, x):
        # Flattening logic for H-rep [Batch, 6, 5, 3] -> [Batch, 6, 15]
        if len(x.shape) == 4:
            batch_size, num_pieces, n_cons, n_params = x.shape
            x = x.view(batch_size, num_pieces, -1)
        
        batch_size, num_pieces, feat_dim = x.shape
        x_flat = x.reshape(-1, feat_dim) 
        local_features = self.phi(x_flat)
        return local_features.view(batch_size, num_pieces, -1)

class DeepSetActorCritic(nn.Module):
    def __init__(self, input_dim, num_pieces, num_actions, hidden_dim=128):
        super().__init__()
        # Shared Feature Extractor
        self.encoder = DeepSetEncoder(input_dim, hidden_dim)
        
        # Latent space after aggregation
        self.rho = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Policy Head (Actor): [Batch, num_actions]
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_actions)
        )
        
        # Value Head (Critic): [Batch, 1]
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        # 1. Local Encoding: Process each polytope piece independently
        local_feats = self.encoder(x) # [Batch, 6, 128]
        
        # 2. Global Aggregation: Sum is permutation invariant
        # f(X) = rho(sum(phi(x_i)))
        global_sum = torch.sum(local_feats, dim=1) # [Batch, 128]
        combined = self.rho(global_sum)
        
        # 3. Heads
        logits = self.actor(combined)
        value = self.critic(combined)
        
        return logits, value