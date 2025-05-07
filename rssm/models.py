
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer
            

class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        in_channels = 3
        embed_dim = 512
        self.encoder = nn.Sequential(
            layer_init(nn.Conv2d(in_channels, 32, 8, stride=4)), #TODO: mudar valor de in channels
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, embed_dim)),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential( #TODO: colocar valor de estados estocasticos e deterministicos
            layer_init(nn.Linear(embed_dim, 64 * 7 * 7)),
            nn.ReLU(),
            nn.Unflatten(1, (64, 7, 7)),  
            layer_init(nn.ConvTranspose2d(64, 64, 3, stride=1)),  # Inverso do último conv
            nn.ReLU(),
            layer_init(nn.ConvTranspose2d(64, 32, 4, stride=2)),  # Inverso do segundo conv
            nn.ReLU(),
            layer_init(nn.ConvTranspose2d(32, in_channels, 8, stride=4)), 
            nn.Sigmoid(),  #TODO: saida normalizada para imagens
        )

        self.actor = layer_init(nn.Linear(embed_dim, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(embed_dim, 1), std=1)

    def get_value(self, x):
        return self.critic(self.encoder(x / 255.0))

    def get_action_and_value(self, x, action=None):
        hidden = self.encoder(x / 255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)

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

class RSSM:
    def __init__(self,
                 agent: Agent,
                 reward_model: RewardModel,
                 dynamics_model: nn.Module,
                 hidden_dim: int,
                 state_dim: int,
                 action_dim: int,
                 embedding_dim: int,
                 device: str = "mps"):
        """
        Recurrent State-Space Model (RSSM) for learning dynamics models.

        Args:
            encoder: Encoder network for deterministic state
            prior_model: Prior network for stochastic state
            decoder: Decoder network for reconstructing observation
            sequence_model: Recurrent model for deterministic state
            hidden_dim: Hidden dimension of the RNN
            latent_dim: Latent dimension of the stochastic state
            action_dim: Dimension of the action space
            obs_dim: Dimension of the encoded observation space


        """
        super(RSSM, self).__init__()

        self.dynamics = dynamics_model
        self.encoder = agent.encoder
        self.decoder = agent.decoder
        self.reward_model = reward_model

        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.embedding_dim = embedding_dim

        #shift to device
        self.dynamics.to(device)
        self.encoder.to(device)
        self.decoder.to(device)
        self.reward_model.to(device)


    def generate_rollout(self, actions: torch.Tensor, hiddens: torch.Tensor = None, states: torch.Tensor = None,
                         obs: torch.Tensor = None, dones: torch.Tensor = None):

        if hiddens is None:
            hiddens = torch.zeros(actions.size(0), self.hidden_dim).to(actions.device)

        if states is None:
            states = torch.zeros(actions.size(0), self.state_dim).to(actions.device)

        dynamics_result = self.dynamics(hiddens, states, actions, obs, dones)
        hiddens, prior_states, posterior_states, prior_means, prior_logvars, posterior_means, posterior_logvars = dynamics_result

        return hiddens, prior_states, posterior_states, prior_means, prior_logvars, posterior_means, posterior_logvars

    def train(self):
        self.dynamics.train()
        self.encoder.train()
        self.decoder.train()
        self.reward_model.train()

    def eval(self):
        self.dynamics.eval()
        self.encoder.eval()
        self.decoder.eval()
        self.reward_model.eval()

    def encode(self, obs: torch.Tensor):
        return self.encoder(obs)

    def decode(self, state: torch.Tensor):
        return self.decoder(state)

    def predict_reward(self, h: torch.Tensor, s: torch.Tensor):
        return self.reward_model(h, s)

    def parameters(self):
        return list(self.dynamics.parameters()) + list(self.encoder.parameters()) + list(self.decoder.parameters()) + list(self.reward_model.parameters())

    def save(self, path: str):
        torch.save({
            "dynamics": self.dynamics.state_dict(),
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
            "reward_model": self.reward_model.state_dict()
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path)
        self.dynamics.load_state_dict(checkpoint["dynamics"])
        self.encoder.load_state_dict(checkpoint["encoder"])
        self.decoder.load_state_dict(checkpoint["decoder"])
        self.reward_model.load_state_dict(checkpoint["reward_model"])

if __name__ == "__main__":
    from types import SimpleNamespace

    # Parâmetros do mock
    batch_size = 4
    image_shape = (3, 84, 84)
    action_dim = 6
    embed_dim = 512
    hidden_dim = 200
    state_dim = 30
    belief_size = 100
    num_layers = 2

    # Mock do ambiente com espaço de ações
    envs = SimpleNamespace(single_action_space=SimpleNamespace(n=action_dim))

    # Tensores simulados
    dummy_obs = torch.randint(0, 256, (batch_size, *image_shape), dtype=torch.uint8).float()  # [4, 3, 84, 84]
    dummy_action = torch.randn(batch_size, action_dim)  # [4, 6]
    dummy_state = torch.randn(batch_size, state_dim)    # [4, 30]
    dummy_hidden = torch.randn(batch_size, hidden_dim)  # [4, 200]
    dummy_belief = torch.randn(batch_size, belief_size) # [4, 100]

    # Instancia os modelos
    agent = Agent(envs)
    reward_model = RewardModel(hidden_dim=hidden_dim, state_dim=state_dim)
    transition_model = TransitionModel(
        latent_dim=state_dim,
        belief_size=belief_size,
        hidden_size=hidden_dim,
        future_rnn=True,
        mean_only=False,
        min_stddev=0.1,
        num_layers=num_layers
    )
    rssm = RSSM(
        agent=agent,
        reward_model=reward_model,
        dynamics_model=transition_model,
        hidden_dim=hidden_dim,
        state_dim=state_dim,
        action_dim=action_dim,
        embedding_dim=embed_dim,
        device="cpu"
    )

    # Testa encoder
    encoded = rssm.encode(dummy_obs)  # Esperado: [4, 512]
    print("Encoded shape:", encoded.shape)  # [4, 512]
    # assert encoded.shape == (4, 512)

    # Testa decoder
    decoded = rssm.decode(encoded)  # Esperado: [4, 3, 84, 84]
    print("Decoded shape:", decoded.shape)
    # assert decoded.shape == (4, 3, 84, 84)

    # Testa agente (ação e valor)
    action, log_prob, entropy, value = agent.get_action_and_value(dummy_obs)
    print("Sampled action:", action.shape)        # Esperado: [4]
    print("Log prob:", log_prob.shape)            # Esperado: [4]
    print("Entropy:", entropy.shape)              # Esperado: [4]
    print("Value shape:", value.shape)            # Esperado: [4, 1]
    # assert action.shape == (4,)
    # assert log_prob.shape == (4,)
    # assert entropy.shape == (4,)
    # assert value.shape == (4, 1)

    # Testa reward model
    reward_logits = rssm.predict_reward(dummy_hidden, dummy_state)  # Input: [4, 200] + [4, 30]
    print("Reward model output shape:", reward_logits.shape)        # Esperado: [4, 2]
    # assert reward_logits.shape == (4, 2)

    # Testa transition model (posterior)
    state_out = transition_model._posterior(
        prev_state={
            'sample': dummy_state,        # [4, 30]
            'rnn_state': dummy_belief,    # [4, 100]
            'belief': dummy_belief,       # [4, 100] ← CORRIGIDO
            'mean': dummy_state,
            'stddev': torch.ones_like(dummy_state)
        },
        prev_action=dummy_action,         # [4, 6]
        obs=torch.randn(batch_size, embed_dim)  # [4, 512]
    )
    print("Transition posterior sample shape:", state_out['sample'].shape)  # Esperado: [4, 30]
    print("Belief shape:", state_out['belief'].shape)                       # Esperado: [4, 100]
    # assert state_out['sample'].shape == (4, 30)
    # assert state_out['belief'].shape == (4, 100)


