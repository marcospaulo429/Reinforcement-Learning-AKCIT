import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
import torch.distributions as distributions
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
import gym
import os
from dm_control import suite
from dm_control.suite.wrappers import pixels
from auxiliares import converter_cinza, training_device
import wandb

device = training_device()

def reparameterize(z_mean, z_log_var):
    std = torch.exp(0.5 * z_log_var)
    eps = torch.randn_like(std)
    return z_mean + eps * std

# Encoder com mais camadas convolucionais para maior poder representacional
class CNNEncoder(nn.Module):
    def __init__(self, latent_dim, in_channels=1,hidden_units = 32):  
        super(CNNEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_units, kernel_size=4, stride=2, padding=1)   # -> (batch, 64, 42, 42)
        self.conv2 = nn.Conv2d(hidden_units, hidden_units*2, kernel_size=4, stride=2, padding=1)            # -> (batch, 128, 21, 21)
        self.conv3 = nn.Conv2d(hidden_units*2, hidden_units*4, kernel_size=4, stride=2, padding=1)           # -> (batch, 256, 10, 10)
        
        self.flatten_dim = hidden_units*4 * 10 * 10  # Tamanho do vetor achatado após as convoluções
        self.fc_mean = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_log_var = nn.Linear(self.flatten_dim, latent_dim)
    
    def forward(self, x):
        x = f.relu(self.conv1(x))
        x = f.relu(self.conv2(x))
        x = f.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Achata para (batch, flatten_dim)
        z_mean = self.fc_mean(x)
        z_log_var = self.fc_log_var(x)
        return z_mean, z_log_var

class CNNDecoder(nn.Module):
    def __init__(self, latent_dim, out_channels=1, hidden_units=32):  # Alterei o out_channels para 3 para imagens RGB
        super(CNNDecoder, self).__init__()
        self.hidden_units = hidden_units
        self.fc = nn.Linear(latent_dim, hidden_units*4 * 10 * 10)  # Mapeia o vetor latente para uma representação plana
        self.deconv1 = nn.ConvTranspose2d(hidden_units*4, hidden_units*2, kernel_size=4, stride=2, padding=1, output_padding=1)  
        self.deconv2 = nn.ConvTranspose2d(hidden_units*2, hidden_units, kernel_size=4, stride=2, padding=1, output_padding=0)
        self.deconv3 = nn.ConvTranspose2d(hidden_units, out_channels, kernel_size=4, stride=2, padding=1, output_padding=0)
    
    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), (self.hidden_units)*4, 10, 10)  # Reorganiza para (batch, 256, 10, 10)
        x = f.relu(self.deconv1(x))
        x = f.relu(self.deconv2(x))
        x = torch.tanh(self.deconv3(x))  # Sigmoid para valores entre 0 e 1
        return x


# VAE combinando o Encoder e o Decoder com CNNs mais complexas
class VAE(nn.Module):
    def __init__(self, latent_dim, in_channels=1, hidden_units=32):
        super(VAE, self).__init__()
        self.encoder = CNNEncoder(latent_dim, in_channels,hidden_units=hidden_units)
        self.decoder = CNNDecoder(latent_dim, out_channels=in_channels, hidden_units=hidden_units)
    
    def forward(self, x):
        z_mean, z_log_var = self.encoder(x)
        z = reparameterize(z_mean, z_log_var)
        recon = self.decoder(z)
        return recon, z_mean, z_log_var


# 1. Defina os mesmos parâmetros usados no treinamento
latent_dim = 128  # Substitua pelo valor usado originalmente
in_channels = 1   # Substitua pelo valor usado originalmente
hidden_units = 128 # Substitua pelo valor usado originalmente

# 2. Instancie o modelo e otimizador (igual ao do treinamento)
model = VAE(latent_dim=latent_dim, in_channels=in_channels, hidden_units=hidden_units).to(device)
optimizer = torch.optim.Adam(model.parameters())  # Ou o otimizador que você usou

# 3. Caminho para o checkpoint
checkpoint_dir = "checkpoint_dgx_treino3-lat128_lr0.0001_h128_epoch570.pt" #TODO
epoch_to_load = 570                           # Epoch que deseja carregar

checkpoint_path = os.path.join(checkpoint_dir)

# 4. Carregue o checkpoint
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
    
    # Carregue os estados
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Outras informações do checkpoint
    loaded_epoch = checkpoint['epoch']
    loaded_loss = checkpoint['loss']
    loaded_test_loss = checkpoint['test_loss']
    
    print(f"Checkpoint carregado com sucesso (Epoch {loaded_epoch})")
    print(f"Loss de treino no checkpoint: {loaded_loss:.4f}")
    print(f"Loss de teste no checkpoint: {loaded_test_loss:.4f}")
    
    # Coloque o modelo em modo de avaliação
    model.eval()
else:
    print(f"Erro: Arquivo de checkpoint não encontrado em {checkpoint_path}")
    

encoder = model.encoder

env_train = suite.load(domain_name="cartpole", task_name="swingup")
env_train = pixels.Wrapper(env_train, pixels_only=True,
                    render_kwargs={'height': 84, 'width': 84, 'camera_id': 0})
# Teste com imagens do ambiente
env_test = suite.load(domain_name="cartpole", task_name="swingup")
env_test = pixels.Wrapper(env_train, pixels_only=True,
                    render_kwargs={'height': 84, 'width': 84, 'camera_id': 0})


class Critic(nn.Module): 
    def __init__(self, in_features, hidden_dimensions, out_features, dropout):
        super().__init__()

        self.layer1 = nn.Linear(in_features, hidden_dimensions)
        self.layer2 = nn.Linear(hidden_dimensions, hidden_dimensions)
        self.layer3 = nn.Sigmoide(hidden_dimensions, out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer1(x)
        x = f.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = f.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = f.relu(x)
        x = self.dropout(x)
        x = self.layer3(x)
        return x


class Actor(nn.Module):
    def __init__(self, in_features, hidden_dimensions, out_features, dropout):
        super().__init__()

        self.layer1 = nn.Linear(in_features, hidden_dimensions)
        self.layer2 = nn.Linear(hidden_dimensions, hidden_dimensions)
        self.layer3 = nn.Linear(hidden_dimensions, out_features)
        self.dropout = nn.Dropout(dropout)
        self.Softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.layer1(x)
        x = f.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = f.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = f.relu(x)
        x = self.dropout(x)
        x = self.layer3(x)
        action_prob = self.Softmax(x)
        dist = distributions.Categorical(action_prob)
        action = dist.sample()
        log_prob_action = dist.log_prob(action)
        return action, log_prob_action, dist


def create_agent(HIDDEN_DIMENSIONS, DROPOUT, INPUT_FEATURES, ACTOR_OUTPUT_FEATURES):
    CRITIC_OUTPUT_FEATURES = 1

    actor = Actor(
            INPUT_FEATURES, HIDDEN_DIMENSIONS, ACTOR_OUTPUT_FEATURES, DROPOUT).to(device)
    critic = Critic(
            INPUT_FEATURES, HIDDEN_DIMENSIONS, CRITIC_OUTPUT_FEATURES, DROPOUT).to(device)
    return actor, critic

def calculate_returns(rewards, discount_factor):
    global device
    returns = []
    cumulative_reward = 0
    for r in reversed(rewards):
        cumulative_reward = r + cumulative_reward * discount_factor
        returns.insert(0, cumulative_reward)

    returns = torch.tensor(returns, device=device)  
    # normalize the return
    returns = (returns - returns.mean()) / returns.std()

    return returns

def calculate_advantages(returns, values):
    advantages = returns - values
    # Normalize the advantage
    advantages = (advantages - advantages.mean()) / advantages.std()
    return advantages

def calculate_surrogate_loss(
        actions_log_probability_old,
        actions_log_probability_new,
        epsilon,
        advantages):
    advantages = advantages.detach()
    policy_ratio = (
            actions_log_probability_new - actions_log_probability_old
            ).exp()
    surrogate_loss_1 = policy_ratio * advantages
    surrogate_loss_2 = torch.clamp(
            policy_ratio, min=1.0-epsilon, max=1.0+epsilon
            ) * advantages
    surrogate_loss = torch.min(surrogate_loss_1, surrogate_loss_2)
    return surrogate_loss

def calculate_losses(
        surrogate_loss, entropy, entropy_coefficient, returns, value_pred):
    entropy_bonus = entropy_coefficient * entropy
    policy_loss = -(surrogate_loss + entropy_bonus).sum()
    value_loss = f.smooth_l1_loss(returns, value_pred).sum()
    return policy_loss, value_loss


def init_training():
    states = []
    actions = []
    actions_log_probability = []
    values = []
    rewards = []
    done = False
    episode_reward = 0
    return states, actions, actions_log_probability, values, rewards, done, episode_reward

def forward_pass(env, actor, critic, encoder, discount_factor):
    states, actions, actions_log_probability, values, rewards, done, episode_reward = init_training()
    time_step = env.reset()

    # Processamento do estado inicial com float32
    state = converter_cinza(time_step.observation['pixels'])
    state = state.astype(np.float32) / 127.5 - 1.0  # Já garante float32
    state = torch.tensor(state, dtype=torch.float32).to(device)
    state = state.unsqueeze(0).unsqueeze(0)
    
    with torch.no_grad():
        z_mean, z_log_var = encoder(state)

    state = reparameterize(z_mean, z_log_var).detach()

    actor.train()
    critic.train()
    while not done:
        states.append(state.detach())
        
        # Garante float32 nas saídas para trabalhar com mac
        action, log_prob_action, _ = actor(state.detach())
        value_pred = critic(state).float() 
        
        time_step = env.step(action.item())
        done = time_step.last()
        reward = time_step.reward if time_step.reward is not None else 0.0
        
        # Processamento do novo estado
        state = converter_cinza(time_step.observation['pixels'])
        state = state.astype(np.float32) / 127.5 - 1.0
        state = torch.tensor(state, dtype=torch.float32).to(device)
        state = state.unsqueeze(0).unsqueeze(0)
        
        with torch.no_grad():
            z_mean, z_log_var = encoder(state)
        state = reparameterize(z_mean, z_log_var)

        actions.append(action)
        actions_log_probability.append(log_prob_action)
        values.append(value_pred)
        rewards.append(float(reward))  # Garante Python float
        episode_reward += reward

    # Concatenação com float32
    states = torch.cat(states).to(device)
    actions = torch.cat(actions).to(device)
    actions_log_probability = torch.cat(actions_log_probability).to(device)
    values = torch.cat(values).squeeze(-1).to(device)
    
    # Modificação crítica: Garante float32 nos retornos
    returns = calculate_returns(rewards, discount_factor).float().to(device)  # <-- Correção aqui
    advantages = calculate_advantages(returns, values).float().to(device)

    return episode_reward, states, actions, actions_log_probability, advantages, returns

def evaluate(env, actor, encoder, device):
    actor.eval()
    
    time_step = env.reset()
    episode_reward = 0
    done = False
    
    # Processamento inicial do estado
    state = converter_cinza(time_step.observation['pixels'])
    state = state.astype(np.float32) / 127.5 - 1.0
    state = torch.tensor(state).float().to(device)  # Convertido para float e movido para o device
    state = state.unsqueeze(0).unsqueeze(0)
    
    with torch.no_grad():
        z_mean, z_log_var = encoder(state)
    state = reparameterize(z_mean, z_log_var).detach()

    while not done:
        with torch.no_grad():
            action, log_prob_action, dist = actor(state)

        time_step = env.step(action.item())
        done = time_step.last()
        reward = time_step.reward if time_step.reward is not None else 0.0
        
        # Processamento do novo estado
        state = converter_cinza(time_step.observation['pixels'])
        state = state.astype(np.float32) / 127.5 - 1.0
        state = torch.tensor(state).float().to(device)  # Convertido para float e movido para o device
        state = state.unsqueeze(0).unsqueeze(0)
        
        with torch.no_grad():
            z_mean, z_log_var = encoder(state)
        state = reparameterize(z_mean, z_log_var).detach()
        
        episode_reward += reward

    return episode_reward

def update_policy(
        actor,
        critic,
        states,
        actions,
        actions_log_probability_old,
        advantages,
        returns,
        optimizer_actor,
        optimizer_critic,
        ppo_steps,
        epsilon,
        entropy_coefficient):

    BATCH_SIZE = 128
    total_policy_loss = 0
    total_value_loss = 0
    actions_log_probability_old = actions_log_probability_old.detach()
    actions = actions.detach()

    training_results_dataset = TensorDataset(
            states,
            actions,
            actions_log_probability_old,
            advantages,
            returns)

    batch_dataset = DataLoader(
            training_results_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False)

    for _ in range(ppo_steps):
        for batch_idx, (states, actions, actions_log_probability_old, advantages, returns) in enumerate(batch_dataset):
            # get new log prob of actions for all input states
            action, actions_log_probability_new, probability_distribution_new = actor(states) 
            value_pred = critic(states)
            value_pred = value_pred.squeeze(-1)
            entropy = probability_distribution_new.entropy()

            # estimate new log probabilities using old actions
            actions_log_probability_new = probability_distribution_new.log_prob(actions)
            surrogate_loss = calculate_surrogate_loss(
                    actions_log_probability_old,
                    actions_log_probability_new,
                    epsilon,
                    advantages)
            policy_loss, value_loss = calculate_losses(
                    surrogate_loss,
                    entropy,
                    entropy_coefficient,
                    returns,
                    value_pred)

            optimizer_actor.zero_grad() 
            optimizer_critic.zero_grad() 
            policy_loss.backward()
            value_loss.backward()
            optimizer_actor.step()
            optimizer_critic.step()
            # calculate the total loss
            # and add it to the total loss
            # for the current batch

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()

    return total_policy_loss / ppo_steps, total_value_loss / ppo_steps

def run_ppo():
    device =  training_device()
    LATENT_DIM = 128
    IN_CHANNELS = 1
    HIDDEN_UNITS = 128
    INPUT_FEATURES = LATENT_DIM
    ACTOR_OUTPUT_FEATURES = 2 
    MAX_EPISODES = 1000
    DISCOUNT_FACTOR = 0.99
    REWARD_THRESHOLD = 500
    PRINT_INTERVAL = 10 #TODO
    PPO_STEPS = 10
    N_TRIALS = 100
    EPSILON = 0.2
    ENTROPY_COEFFICIENT = 0.01

    HIDDEN_DIMENSIONS = 32
    DROPOUT = 0.2
    LEARNING_RATE_ACTOR = 0.001
    LEARNING_RATE_CRITIC = 0.001

    train_rewards = []
    test_rewards = []
    policy_losses = []
    value_losses = []
    name_wandb = "ppo_dreamer2"

    wandb.init(project="ppo_dreamer", name= name_wandb)

    actor, critic = create_agent(HIDDEN_DIMENSIONS, DROPOUT, INPUT_FEATURES, ACTOR_OUTPUT_FEATURES)
    optimizer_actor = optim.Adam(actor.parameters(), lr=LEARNING_RATE_ACTOR)
    optimizer_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE_CRITIC)

    for episode in range(1, MAX_EPISODES+1):
        train_reward, states, actions, actions_log_probability, advantages, returns = forward_pass(
                env_train,
                actor,
                critic,
                encoder,
                DISCOUNT_FACTOR)
        policy_loss, value_loss = update_policy(
                actor,
                critic,
                states,
                actions,
                actions_log_probability,
                advantages,
                returns,
                optimizer_actor,
                optimizer_critic,
                PPO_STEPS,
                EPSILON,
                ENTROPY_COEFFICIENT)
        test_reward = evaluate(env_test, actor, encoder, device)

        wandb.log({
            "episode": episode,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "train_rewards": train_reward,
            "test_rewards": test_reward  # Loga as 20 imagens de teste
        })

        policy_losses.append(policy_loss)
        value_losses.append(value_loss)
        train_rewards.append(train_reward)
        test_rewards.append(test_reward)

        if episode % PRINT_INTERVAL == 0:
            print(f'Episode: {episode:3d} | Training Reward: {train_reward:.1f} | '
                  f'Testing Reward: {test_reward:.1f} | '
                  f' Policy Loss: {policy_loss:.2f} | '
                  f'Value Loss: {value_loss:.2f} | ')


run_ppo() 