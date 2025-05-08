
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class RewardModel(nn.Module):
    def __init__(self, hidden_dim: int, state_dim: int):
        super(RewardModel, self).__init__()

        self.fc1 = nn.Linear(hidden_dim + state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 2)

    def forward(self, h: torch.Tensor, s: torch.Tensor):
        x = torch.cat([h, s], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)

        return x

class TransitionModel(nn.Module):
    def __init__(self, latent_dim, belief_size, hidden_size, future_rnn, mean_only, min_stddev, num_layers):
        super().__init__()
        self.latent_dim = latent_dim
        self.belief_size = belief_size
        self.hidden_size = hidden_size
        self.gru = nn.GRUCell(input_size=hidden_size, hidden_size=belief_size)
        self.future_rnn = future_rnn
        self.mean_only = mean_only
        self.min_stddev = min_stddev
        self.num_layers = num_layers

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
        # 1. Concatenar estado latente e ação
        hidden = torch.cat([prev_state['sample'], prev_action], dim=-1)
        
        # 2. Camadas densas pré-RNN (usando hidden_size)
        for _ in range(self.num_layers):
            hidden = nn.Linear(hidden.size(-1), self.hidden_size)(hidden)
            hidden = nn.ELU()(hidden)  # Ativação ELU padrão
        
        # 3. Passar pela GRU (atenção à dimensão de entrada!)
        # O GRUCell espera input_size = hidden_size (ajustamos na concatenação)
        belief = self.gru(hidden, prev_state['rnn_state'])
        
        # 4. Camadas pós-RNN (se future_rnn=True)
        if self.future_rnn:
            hidden = belief

        for _ in range(self.num_layers):
            hidden = nn.Linear(hidden.size(-1), self.hidden_size)(hidden)
            hidden = nn.ELU()(hidden)
        
        # 5. Calcular média e desvio padrão
        mean = nn.Linear(hidden.size(-1), self.latent_dim)(hidden)
        stddev = F.softplus(nn.Linear(hidden.size(-1), self.latent_dim)(hidden)) + self.min_stddev
        
        # 6. Amostrar ou usar média direta
        sample = mean if self.mean_only else torch.distributions.Normal(mean, stddev).sample()
        
        return {
            'mean': mean,
            'stddev': stddev,
            'sample': sample,
            'belief': belief,
            'rnn_state': belief  # Em GRUCell, o novo estado é a própria saída TODO
        }
    
    def _posterior(self, prev_state, prev_action, obs): 
        prior = self._transition(prev_state, prev_action)
        hidden = torch.concat([prior['belief'], obs], -1)

        for _ in range(self.num_layers):
            hidden = nn.Linear(hidden.size(-1), self.hidden_size)(hidden)
            hidden = nn.ELU()(hidden)  

        mean = nn.Linear(hidden.size(-1), self.latent_dim)(hidden)
        stddev = F.softplus(nn.Linear(hidden.size(-1), self.latent_dim)(hidden)) + self.min_stddev

        sample = mean if self.mean_only else torch.distributions.Normal(mean, stddev).sample()

        return {
            'mean': mean,
            'stddev': stddev,
            'sample': sample,
            'belief': prior['belief'],
            'rnn_state': prior['rnn_state'],
        }

    def imagine_traj(self, obs, horizon, prev_action, actor, reward_model, value_model):
        """
        Gera uma trajetória imaginada no espaço latente a partir de uma observação.
        
        :param obs: estado observado atual (codificado) [B, obs_dim]
        :param horizon: número de passos a imaginar
        :param prev_action: ação anterior [B, action_dim]
        :param actor: rede de política que age no espaço latente
        :param reward_model: rede que estima recompensa latente
        :param value_model: rede que estima valor latente
        """
        B = obs.size(0)
        device = obs.device

        traj = []
        rewards = []
        values = []

        # Inicializar com o posterior (estado real)
        state = self._posterior(
            prev_state={'sample': torch.zeros(B, self.latent_dim, device=device),
                        'rnn_state': torch.zeros(B, self.hidden_size, device=device)},
            prev_action=prev_action,
            obs=obs
        )

        for t in range(horizon):
            # 1. Escolher ação latente com a política
            action = actor(state['sample'].detach(), state['belief'].detach()) #TODO: fazer detach?

            # 2. Fazer transição no modelo (imaginação)
            state = self._transition(state, action)

            # 3. Prever recompensa e valor no espaço latente
            reward = reward_model(state['sample'])
            value = value_model(state['sample'])

            # 4. Armazenar
            traj.append(state)
            rewards.append(reward)
            values.append(value)

        # Agrupar listas em tensores
        traj_dict = {
            'states': {k: torch.stack([s[k] for s in traj], dim=0) for k in traj[0]},
            'rewards': torch.stack(rewards, dim=0),
            'values': torch.stack(values, dim=0),
            'actions': torch.stack([actor(s['sample'], s['belief']) for s in traj], dim=0)
        }

        return traj_dict

