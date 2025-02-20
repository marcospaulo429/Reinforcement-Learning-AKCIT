import torch
import torch.nn as nn

class Actor(nn.Module):
    def __init__(self, latent_dim, action_dim):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()  # ações em [-1,1]
        )
    def forward(self, latent):
        return self.net(latent)

class ValueNet(nn.Module):
    def __init__(self, latent_dim):
        super(ValueNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    def forward(self, latent):
        return self.net(latent)
