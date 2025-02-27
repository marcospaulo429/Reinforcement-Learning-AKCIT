import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvEncoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=1024):
        super().__init__()
        # Exemplo de arquitetura de encoder convolucional
        self.conv_net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2), nn.ReLU()
        )
        # Camada linear para produzir o vetor latente final
        self.fc = nn.Linear(256 * 4 * 4, latent_dim)  # assumindo entrada 64x64

    def forward(self, obs):
        """Recebe uma observação de imagem e retorna um vetor latente."""
        x = obs / 255.0  # normalizar pixel 0-1
        features = self.conv_net(x)              # extração convolucional
        features = features.view(features.size(0), -1) 
        latent = self.fc(features)              # projeção para vetor latente
        return latent

class ConvDecoder(nn.Module):
    def __init__(self, latent_dim=1024, out_channels=3):
        super().__init__()
        # Projeta de volta para mapa de ativação inicial
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)
        # Sequência de camadas de deconvolução para gerar imagem
        self.deconv_net = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2, output_padding=0), nn.ReLU(),
            nn.ConvTranspose2d(32, out_channels, kernel_size=6, stride=2, output_padding=0)
            # A última camada normalmente usa ativação sigmoid para limitar [0,1]
        )

    def forward(self, latent):
        """Recebe um vetor latente e reconstrói a imagem."""
        x = self.fc(latent)
        x = x.view(x.size(0), 256, 4, 4)  # reshape para tensor 4x4 com 256 canais
        recon = self.deconv_net(x)
        recon = torch.sigmoid(recon)  # saída normalizada 0-1
        return recon

class RSSM(nn.Module):
    def __init__(self, latent_dim, deter_dim, action_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.deter_dim = deter_dim
        # Módulo recorrente determinístico (GRU)
        self.rnn = nn.GRUCell(input_size=latent_dim + action_dim, hidden_size=deter_dim)
        # Redes densas auxiliares para prior e posterior (3 camadas de 300 unidades ELU)
        def build_mlp(input_dim, output_dim):
            return nn.Sequential(
                nn.Linear(input_dim, 300), nn.ELU(),
                nn.Linear(300, 300), nn.ELU(),
                nn.Linear(300, 300), nn.ELU(),
                nn.Linear(300, output_dim)
            )
        # Prior recebe apenas estado determinístico h_t como entrada
        self.prior_net = build_mlp(deter_dim, 2 * latent_dim)  # outputs [mean, logvar]
        # Posterior recebe estado determinístico concatenado com embedding da obs
        self.post_net = build_mlp(deter_dim + encoder_out_dim, 2 * latent_dim)
        # Nota: encoder_out_dim é a dimensão do vetor latente do encoder de observação
        # (p.ex., 1024 se for o tamanho do vetor latente do ConvEncoder)
    
    def init_state(self, batch_size):
        """Inicializa estado oculto determinístico (h) e estocástico (z) iniciais."""
        h0 = torch.zeros(batch_size, self.deter_dim)
        z0 = torch.zeros(batch_size, self.latent_dim)
        return (h0, z0)
    
    def forward(self, prev_state, prev_action, obs_embed=None):
        """
        Avança o RSSM de um passo. 
        Se obs_embed for fornecido, calcula posterior (treinamento); 
        caso contrário, usa apenas prior (imaginação).
        """
        h_prev, z_prev = prev_state  # estado anterior
        # Combina z_prev e ação anterior como entrada para o RNN determinístico
        rnn_input = torch.cat([z_prev, prev_action], dim=-1)
        h_t = self.rnn(rnn_input, h_prev)  # atualiza estado determinístico
        # Prior: usa h_t para prever z_t
        prior_params = self.prior_net(h_t)              # obtém [mean, logvar] do prior
        prior_mean, prior_logvar = prior_params.split(self.latent_dim, dim=-1)
        prior_std = torch.exp(0.5 * prior_logvar)       # desvio padrão via logvar
        prior_dist = torch.distributions.Normal(prior_mean, prior_std)
        if obs_embed is None:
            # Sem observação: amostra z_t do prior (modo imaginação)
            z_t = prior_dist.rsample()  # amostragem reparametrizada
            return (h_t, z_t), prior_dist, None
        else:
            # Posterior: ajusta com evidência da observação
            post_input = torch.cat([h_t, obs_embed], dim=-1)
            post_params = self.post_net(post_input)     # [mean, logvar] do posterior
            post_mean, post_logvar = post_params.split(self.latent_dim, dim=-1)
            post_std = torch.exp(0.5 * post_logvar)
            post_dist = torch.distributions.Normal(post_mean, post_std)
            # Amostra z_t do posterior durante treinamento
            z_t = post_dist.rsample()
            return (h_t, z_t), prior_dist, post_dist

def make_dense_network(input_dim, output_dim):
    """Cria uma rede densa de 3 camadas ocultas (300 neurônios, ELU) com dimensão de saída especificada."""
    return nn.Sequential(
        nn.Linear(input_dim, 300), nn.ELU(),
        nn.Linear(300, 300), nn.ELU(),
        nn.Linear(300, 300), nn.ELU(),
        nn.Linear(300, output_dim)
    )

# Exemplo: Rede de recompensa que recebe estado latente combinado (h_t concat z_t) e produz escalar
class RewardModel(nn.Module):
    def __init__(self, latent_dim=30, deter_dim=200):
        super().__init__()
        self.net = make_dense_network(latent_dim + deter_dim, 1)  # output 1 valor de recompensa

    def forward(self, h, z):
        x = torch.cat([h, z], dim=-1)
        return self.net(x)


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
        self.fc1 = nn.Linear(latent_dim, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 200)
        self.value_layer = nn.Linear(200, 1)

    def forward(self, latent):
        x = F.elu(self.fc1(latent))
        x = F.elu(self.fc2(x))
        x = F.elu(self.fc3(x))
        value = self.value_layer(x)
        return value
