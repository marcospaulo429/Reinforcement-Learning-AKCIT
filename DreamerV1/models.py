
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical



def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Encoder(nn.Module):
    def __init__(self, latent_dim=32):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim

        # Camadas convolucionais
        self.encoder_cnn = nn.Sequential(
            layer_init(nn.Conv2d(4, 16, kernel_size=8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(16, 32, kernel_size=4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 32, kernel_size=3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Camadas fully connected para mu e logvar
        self.encoder_fc_mu = layer_init(nn.Linear(32 * 7 * 7, latent_dim))
        self.encoder_fc_logvar = layer_init(nn.Linear(32 * 7 * 7, latent_dim))

    def forward(self, x):
        x = self.encoder_cnn(x)
        mu = self.encoder_fc_mu(x)
        logvar = self.encoder_fc_logvar(x)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim=32):
        super(Decoder, self).__init__()

        # Camada fully connected
        self.decoder_fc = layer_init(nn.Linear(latent_dim, 32 * 7 * 7))
        
        # Camadas deconvolucionais (transpostas)
        self.decoder_deconv = nn.Sequential(
            nn.ReLU(),
            nn.Unflatten(1, (32, 7, 7)),
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 4, kernel_size=8, stride=4),
            nn.Sigmoid()
        )

    def forward(self, z):
        x = self.decoder_fc(z)
        return self.decoder_deconv(x)

def vae_loss(recon_x, x, mu, logvar):
    recon_loss = nn.functional.binary_cross_entropy(recon_x, x, reduction='mean')
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl

class Agent(nn.Module):
    def __init__(self, latent_dim, envs):
        super().__init__()
        self.actor = layer_init(nn.Linear(latent_dim, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(latent_dim, 1), std=1)

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


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