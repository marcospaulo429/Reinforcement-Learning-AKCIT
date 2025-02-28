import torch
import torch.nn as nn
import torch.nn.functional as F


class ActionModel(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # Rede densa de 3 camadas (300 neurônios ELU cada) para mapear estado latente -> parâmetros de ação
        self.net = nn.Sequential(
            nn.Linear(state_dim, 200), nn.ELU(),
            nn.Linear(200, 200), nn.ELU(),
            nn.Linear(200, 200), nn.ELU()
        )
        # Camadas finais separadas para média e desvio padrão (log-variância) da ação
        self.mean_head = nn.Linear(200, action_dim)
        self.std_head  = nn.Linear(200, action_dim)
        # Escala máxima para ações (fator de saturação, ex: 5)
        self.action_scale = 5.0

    def forward(self, latent_state):
        """
        Calcula a distribuição Gaussiana transformada por tanh para a ação.
        Retorna a distribuição transformada, a média e o desvio padrão.
        """
        hidden = self.net(latent_state)
        raw_mean = self.mean_head(hidden)
        raw_std  = self.std_head(hidden)
        # Limita os valores para evitar extremos que possam gerar NaNs
        raw_mean = torch.clamp(raw_mean, -10.0, 10.0)
        raw_std  = torch.clamp(raw_std, -10.0, 10.0)
        # Média: aplica tanh e escala pelo fator
        mean = torch.tanh(raw_mean) * self.action_scale
        # Desvio padrão: usa softplus para garantir positividade e adiciona um pequeno offset
        std = F.softplus(raw_std) + 1e-4
        # Cria a distribuição Normal e aplica a transformação Tanh
        dist = torch.distributions.Normal(mean, std)
        dist_tanh = torch.distributions.TransformedDistribution(
            dist, 
            torch.distributions.transforms.TanhTransform(cache_size=1)
        )
        return dist_tanh, mean, std


class ValueNet(nn.Module):
    def __init__(self, latent_dim):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 300)
        self.fc2 = nn.Linear(300, 300)
        self.fc3 = nn.Linear(300, 300)
        self.value_layer = nn.Linear(300, 1)

    def forward(self, latent):
        x = F.elu(self.fc1(latent))
        x = F.elu(self.fc2(x))
        x = F.elu(self.fc3(x))
        value = self.value_layer(x)
        return value
