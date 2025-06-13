
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class RewardModel(nn.Module):
    def __init__(self, hidden_dim: int, state_dim: int):
        super(RewardModel, self).__init__()

        self.fc1 = nn.Linear(hidden_dim + state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, h: torch.Tensor, s: torch.Tensor):
        x = torch.cat([h, s], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)

        return x

class TransitionModel(nn.Module):
    def __init__(self, latent_dim, belief_size, hidden_size, future_rnn, action_dim, mean_only, min_stddev, num_layers):
        super().__init__()
        self.latent_dim = latent_dim
        self.belief_size = belief_size
        self.hidden_size = hidden_size
        self.future_rnn = future_rnn
        self.mean_only = mean_only
        self.min_stddev = min_stddev
        self.num_layers = num_layers

        self.gru = nn.GRUCell(input_size=hidden_size, hidden_size=belief_size)

        # Camadas densas antes da GRU
        # Camadas densas antes da GRU
        self.pre_rnn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim + action_dim if i == 0 else hidden_size, hidden_size),  # Corrigido
                nn.ELU()
            ) for i in range(num_layers)
        ])


        # Camadas densas depois da GRU
        self.post_rnn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(belief_size if i == 0 else hidden_size, hidden_size),
                nn.ELU()
            ) for i in range(num_layers)
        ])

        # Camadas para média e desvio padrão
        self.mean_layer = nn.Linear(hidden_size, latent_dim)
        self.std_layer = nn.Linear(hidden_size, latent_dim)

        # Camadas do posterior (obs + belief)
        self.posterior_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(belief_size + latent_dim if i == 0 else hidden_size, hidden_size),
                nn.ELU()
            ) for i in range(num_layers)
        ])
        self.posterior_mean = nn.Linear(hidden_size, latent_dim)
        self.posterior_std = nn.Linear(hidden_size, latent_dim)


    @property
    def state_size(self):
        return {
            'mean': self.hidden_size,
            'stddev': self.hidden_size,
            'sample': self.hidden_size,
            'belief': self.belief_size,
            'rnn_state': self.belief_size,
        }

    @property
    def feature_size(self):
        return self.belief_size + self.hidden_size
    
    def dist_from_state(self, state, mask=None):
        import torch.distributions as dist
        # Processa o desvio padrão com a máscara
        if mask is not None:
            # Preenche valores mascarados com 1.0 para nao afetar no calculo
            stddev = torch.where(mask, state['stddev'], torch.ones_like(state['stddev']))
        else:
            stddev = state['stddev']
        
        # Cria distribuição normal multivariada diagonal
        return dist.Normal(state['mean'], stddev)
    
    def features_from_state(self, state):
        return torch.cat([state['belief'], state['sample']], dim=-1)

    def divergence_from_states(self, lhs, rhs, mask=None):
        lhs = self.dist_from_state(lhs, mask)
        rhs = self.dist_from_state(rhs, mask)
        divergence = torch.distributions.kl_divergence(lhs, rhs)
        if mask is not None:
            divergence = torch.where(mask, divergence, torch.zeros_like(divergence))
        return divergence
    
    def _transition(self, prev_state, prev_action):

        hidden = torch.cat([prev_state['sample'], prev_action], dim=-1)

        for layer in self.pre_rnn_layers:
            hidden = layer(hidden)

        belief = self.gru(hidden, prev_state['rnn_state'])

        hidden = belief if self.future_rnn else hidden
        for layer in self.post_rnn_layers:
            hidden = layer(hidden)

        mean = self.mean_layer(hidden)
        stddev = F.softplus(self.std_layer(hidden)) + self.min_stddev
        sample = mean if self.mean_only else torch.distributions.Normal(mean, stddev).rsample()

        return {
            'mean': mean,
            'stddev': stddev,
            'sample': sample,
            'belief': belief,
            'rnn_state': belief
        }
    
    def _posterior(self, prev_state, prev_action, obs):
        prior = self._transition(prev_state, prev_action)
        hidden = torch.cat([prior['belief'], obs], dim=-1)

        for layer in self.posterior_layers:
            hidden = layer(hidden)

        mean = self.posterior_mean(hidden)
        stddev = F.softplus(self.posterior_std(hidden)) + self.min_stddev
        sample = mean if self.mean_only else torch.distributions.Normal(mean, stddev).rsample()

        return {
            'mean': mean,
            'stddev': stddev,
            'sample': sample,
            'belief': prior['belief'],
            'rnn_state': prior['rnn_state']
        }