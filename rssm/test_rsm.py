
import torch
import torch.nn as nn
import torch.nn.functional as F


class RSSM(nn.Module):
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
    
    def _transition(self, prev_state, prev_action, zero_obs):
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
        prior = self._transition(prev_state, prev_action, torch.zeros_like(obs))
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


# Configuração de teste
latent_dim = 512
belief_size = 64
hidden_size = 128
batch_size = 1
seq_len = 10

# Cria instância do RSSM
rssm = RSSM(latent_dim=latent_dim,
            belief_size=belief_size,
            hidden_size=hidden_size,
            future_rnn=True,
            mean_only=False,
            min_stddev=0.1,
            num_layers=2)

# Estado inicial
initial_state = {
    'mean': torch.randn(batch_size, latent_dim),
    'stddev': torch.rand(batch_size, latent_dim),
    'sample': torch.randn(batch_size, latent_dim),
    'belief': torch.randn(batch_size, belief_size),
    'rnn_state': torch.randn(batch_size, belief_size)
}

# Dados de teste (aleatórios)
test_action = torch.randn(batch_size, 10)  # Ex: 10 dimensões de ação
test_obs = torch.randn(batch_size, 50)    # Ex: 50 dimensões de observação

print("=== Teste de Transição ===")
next_state = rssm._transition(initial_state, test_action, torch.zeros_like(test_obs))
print("Estado resultante:")
for k, v in next_state.items():
    print(f"{k}: {v.shape}")

print("\n=== Teste de Posterior ===")
posterior_state = rssm._posterior(initial_state, test_action, test_obs)
print("Estado posterior:")
for k, v in posterior_state.items():
    print(f"{k}: {v.shape}")

print("\n=== Teste de Features ===")
features = rssm.features_from_state(posterior_state)
print(f"Features shape: {features.shape} (deveria ser {batch_size, belief_size + latent_dim})")

print("\n=== Teste de Divergência KL ===")
divergence = rssm.divergence_from_states(next_state, posterior_state)
print(f"Divergência shape: {divergence.shape} (deveria ser {batch_size, latent_dim})")

print("\n=== Teste com Máscara ===")
mask = torch.randint(0, 2, (batch_size,)).bool()
masked_divergence = rssm.divergence_from_states(next_state, posterior_state, mask.unsqueeze(-1))
print(f"Divergência com máscara shape: {masked_divergence.shape}")



def train_rssm_step(rssm, observations, actions, encoder, decoder, optimizer, latent_dim, kl_scale=1.0):
    """
    Um passo de treino do RSSM com observações reais e ações.

    Args:
        rssm: sua instância do modelo RSSM.
        observations: Tensor [B, T, obs_dim]
        actions: Tensor [B, T, action_dim]
        encoder: função que mapeia obs → embed (aqui pode ser um mock)
        decoder: função que mapeia features → obs_recon
        optimizer: otimizador (Adam, etc)
        kl_scale: fator multiplicador para a perda KL

    Returns:
        dicionário com perdas
    """

    B, T, _ = observations.shape

    # 1. Codificar observações reais
    embed = torch.radn([B,T, latent_dim])  # [B, T, embed_dim]

    # 2. Inicializar estado latente zero
    device = observations.device
    state = {
        'mean': torch.zeros(B, rssm.latent_dim, device=device),
        'stddev': torch.ones(B, rssm.latent_dim, device=device),
        'sample': torch.zeros(B, rssm.latent_dim, device=device),
        'belief': torch.zeros(B, rssm.belief_size, device=device),
        'rnn_state': torch.zeros(B, rssm.belief_size, device=device),
    }

    prior_states = []
    posterior_states = []
    reconstructions = []
    kls = []

    # 3. Loop temporal (scan como no Dreamer)
    for t in range(T):
        action = actions[:, t]
        obs_embed = embed[:, t]

        # Estado prior (sem observação)
        prior = rssm._transition(state, action, zero_obs=None)

        # Estado posterior (com observação real)
        posterior = rssm._posterior(state, action, obs_embed)

        # Reconstrução da observação
        features = torch.cat([posterior['belief'], posterior['sample']], dim=-1)
        recon = decoder(features)

        # KL divergence entre posterior e prior
        kl = rssm.divergence_from_states(posterior, prior).mean()

        prior_states.append(prior)
        posterior_states.append(posterior)
        reconstructions.append(recon)
        kls.append(kl)

        state = posterior  # atualiza para o próximo passo

    # 4. Calcular perdas
    recon_obs = torch.stack(reconstructions, dim=1)  # [B, T, obs_dim]
    total_kl = torch.stack(kls, dim=0).mean()

    recon_loss = F.mse_loss(recon_obs, observations)  # reconstrução da observação

    total_loss = recon_loss + kl_scale * total_kl

    # 5. Backprop e otimização
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return {
        'loss': total_loss.item(),
        'recon_loss': recon_loss.item(),
        'kl_loss': total_kl.item()
    }
