import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent)) 

import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.distributions import Normal, TransformedDistribution
from torch.distributions.transforms import TanhTransform
import numpy as np
from utils.auxiliares import training_device
device = training_device()


class Critic(nn.Module):
    def __init__(self, in_features, hidden_dimensions, out_features, dropout):
        super().__init__()
        self.layer1 = nn.Linear(in_features, hidden_dimensions)
        self.layer2 = nn.Linear(hidden_dimensions, hidden_dimensions)
        self.layer3 = nn.Linear(hidden_dimensions, out_features) 
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer1(x)
        x = f.relu(x)
        #x = self.dropout(x)
        x = self.layer2(x)
        x = f.relu(x)
        #x = self.dropout(x)
        x = self.layer3(x)  
        return x

class Actor(nn.Module):
    def __init__(self, input_dim, hidden_dim, action_dim, num_layers=2, init_std=0.1, min_std=0.01):
        super().__init__()
        self.layers = nn.ModuleList()
        self.action_dim = action_dim
        
        # Camadas ocultas
        for i in range(num_layers):
            in_size = input_dim if i == 0 else hidden_dim
            self.layers.append(nn.Linear(in_size, hidden_dim))
            self.layers.append(nn.LayerNorm(hidden_dim))
        
        # Camada de saída para média e desvio padrão
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)
        
        # Parâmetros para o desvio padrão
        self.min_std = min_std
        self.log_std_layer.weight.data.fill_(0.0) #TODO: por que fazer isso?
        self.log_std_layer.bias.data.fill_(np.log(np.exp(init_std) - 1))  # Softplus inverso

    def forward(self, x):
        # Passagem pelas camadas ocultas
        for layer in self.layers:
            x = f.relu(layer(x))
        
        # Calcula média e desvio padrão
        mean = torch.tanh(self.mean_layer(x))  # [-1, 1]
        log_std = self.log_std_layer(x)
        std = f.softplus(log_std) + self.min_std  # Sempre positivo
        
        # Cria distribuição com transformação tanh
        base_dist = Normal(mean, std)
        transforms = TanhTransform()
        dist = TransformedDistribution(base_dist, transforms)
        action = dist.sample()
        log_prob_action = dist.log_prob(action)
        
        return action, log_prob_action, dist