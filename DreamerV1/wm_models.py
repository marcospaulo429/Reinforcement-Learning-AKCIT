import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt

class Autoencoder(nn.Module):
    def __init__(self, input_size,latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, latent_dim*2),
            nn.ReLU(True),
            nn.Linear(latent_dim*2, latent_dim),  # vetor latente de dimensão 64
            nn.ReLU(True),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim*2),
            nn.ReLU(True),
            nn.Linear(latent_dim*2, input_size),
            nn.Tanh(),  # saída em [-1, 1]
        )
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class TransitionModel(nn.Module):
    def __init__(self, latent_dim, action_dim, hidden_dim):
        super(TransitionModel, self).__init__()
        self.gru = nn.GRU(latent_dim + action_dim, hidden_dim, batch_first=True)
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_std  = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, prev_hidden, latent, action):
        x = torch.cat([latent, action], dim=-1).unsqueeze(1)  # [batch, 1, latent_dim+action_dim]
        output, hidden = self.gru(x, prev_hidden.unsqueeze(0))
        hidden = hidden.squeeze(0)  # [batch, hidden_dim]
        mean = self.fc_mean(hidden)
        std  = torch.exp(self.fc_std(hidden))
        eps = torch.randn_like(std)
        latent_next = mean + eps * std
        return latent_next, hidden, mean, std

class RewardModel(nn.Module):
    def __init__(self, latent_dim):
        super(RewardModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    def forward(self, latent):
        return self.fc(latent)