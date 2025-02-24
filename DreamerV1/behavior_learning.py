import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym


from dm_control import suite
from dm_control.suite.wrappers import pixels

from replay_buffer import ReplayBuffer
from world_model import DreamerWorldModel, converter_cinza, get_data_loaders_from_replay_buffer, ver_reconstrucoes, collect_replay_buffer, train_world_model
from torch.utils.tensorboard import SummaryWriter

class Actor(nn.Module):
    def __init__(self, latent_dim, action_dim):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            #nn.Tanh()  # ações em [-1,1]
        )
    def forward(self, latent):
        return self.net(latent)

class ValueNet(nn.Module):
    def __init__(self, latent_dim):
        super(ValueNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    def forward(self, latent):
        return self.net(latent)
    
def imagine_rollout(world_model, actor, initial_latent, initial_hidden, horizon=5, gamma=0.99):
    """
    Gera um rollout imaginado no espaço latente.
    Retorna: latents, rewards e returns, com shape (horizon, B, dim).
    """
    latents = []
    rewards = []
    hidden = initial_hidden
    latent = initial_latent

    for t in range(horizon):
        action = actor(latent)
        latent, hidden, mean, std = world_model.transition_model(hidden, latent, action)
        r = world_model.reward_model(latent)
        latents.append(latent)
        rewards.append(r)
    
    returns = []
    ret = torch.zeros_like(rewards[-1])
    for r in reversed(rewards):
        ret = r + gamma * ret
        returns.insert(0, ret)
    
    latents = torch.stack(latents, dim=0)
    rewards = torch.stack(rewards, dim=0)
    returns = torch.stack(returns, dim=0)
    return latents, rewards, returns
   
def behavior_learning(world_model, actor, value_net, epochs_behavior, train_loader, device,
                      writer, horizon=5, gamma=0.99,
                      value_optimizer=None, actor_optimizer=None,
                      mse_loss=None):
    
    # Inicializa listas para acumular as losses por época
    actor_loss_history = []
    value_loss_history = []
    
    obs, _, _, _ = next(iter(train_loader))
    if obs.dim() == 2 and obs.size(1) == 84 * 84:
        obs = obs.view(obs.size(0), 1, 84, 84)
    obs = obs.to(device)
    batch_obs = obs  
    batch_size = batch_obs.size(0)
    device = batch_obs.device

    hidden_dim = world_model.transition_model.gru.hidden_size

    conv_out = world_model.autoencoder.encoder_conv(obs)  # (B, 64, 21, 21)
    conv_out = conv_out.view(conv_out.size(0), -1)           # (B, 28224)
    latent = world_model.autoencoder.encoder_fc(conv_out)    # (B, latent_dim)

    prev_hidden = torch.zeros(batch_size, hidden_dim, device=device)

    for b_ep in range(epochs_behavior):
        epoch_actor_loss = 0.0
        epoch_value_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            obs, _, _, _ = batch
            obs = obs.to(device)
            if obs.dim() == 2 and obs.size(1) == 84 * 84:
                obs = obs.view(obs.size(0), 1, 84, 84)
            
            batch_size = obs.size(0)
            conv_out = world_model.autoencoder.encoder_conv(obs)  # (B, 64, 21, 21)
            conv_out = conv_out.view(conv_out.size(0), -1)           # (B, 28224)
            latent_init = world_model.autoencoder.encoder_fc(conv_out)   # (B, latent_dim)
            
            prev_hidden = torch.zeros(batch_size, hidden_dim, device=device)
            
            latents, rewards, returns = imagine_rollout(world_model, actor, latent_init, prev_hidden, horizon, gamma)
            
            # Calcula a loss do value_net ao longo do horizonte
            value_loss = 0.0
            for t in range(horizon):
                v_pred = value_net(latents[t])
                value_loss += mse_loss(v_pred, returns[t])
            value_loss = value_loss / horizon

            value_optimizer.zero_grad()
            value_loss.backward(retain_graph=True)
            value_optimizer.step()

            actor_loss = -value_net(latents).mean()
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            epoch_value_loss += value_loss.item()
            epoch_actor_loss += actor_loss.item()
            num_batches += 1

        avg_value_loss = epoch_value_loss / num_batches
        avg_actor_loss = epoch_actor_loss / num_batches

        # Ao invés de escrever no writer, acumulamos os valores
        value_loss_history.append(avg_value_loss)
        actor_loss_history.append(avg_actor_loss)
        print(f"Epoch {b_ep+1}/{epochs_behavior} - Actor Loss: {avg_actor_loss:.4f} | Value Loss: {avg_value_loss:.4f}")

    writer.close()
    return actor, value_net, actor_loss_history, value_loss_history


def select_action(actor, state):
    state = torch.FloatTensor(state).unsqueeze(0)
    logits = actor(state)                     # Saída sem Tanh, logits livres
    action_prob = torch.softmax(logits, dim=-1) # Converte para distribuição de probabilidade
    action = torch.multinomial(action_prob, num_samples=1)
    return action.item(), action_prob[0, action.item()]



def main():
    env = gym.make('CartPole-v1')
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    latent_dim = obs_dim  

    actor = Actor(latent_dim, action_dim)
    value_net = ValueNet(latent_dim)

    actor_optimizer = optim.Adam(actor.parameters(), lr=1e-3)
    value_optimizer = optim.Adam(value_net.parameters(), lr=1e-3)
    mse_loss = nn.MSELoss()
    gamma = 0.99
    num_episodes = 300
    repositorio = "behavior/model_3"

    writer = SummaryWriter(repositorio)

    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        step_count = 0

        while not done:
            step_count += 1

            # 1) Seleciona ação da política
            action, action_prob = select_action(actor, state)
            log_prob = torch.log(action_prob)

            # Executa ação
            next_state, reward, done, _, _ = env.step(action)
            total_reward += reward

            # Converte para tensores
            state_t = torch.FloatTensor(state).unsqueeze(0)
            next_state_t = torch.FloatTensor(next_state).unsqueeze(0)

            # 2) Calcula V(s) e V(s')
            value_s = value_net(state_t)            # Valor atual
            with torch.no_grad():
                value_sp = value_net(next_state_t)  # Valor próximo estado
                if done:  # se episódio acabou, não há valor futuro
                    value_sp = torch.zeros_like(value_sp)

            # TD Target = r + γ V(s')
            td_target = reward + gamma * value_sp

            # 3) Vantagem
            advantages = td_target - value_s
            #normalizacxao pois a loss dos value estava muito alta


            # 4) Perda do ator = -log_prob * advantage
            # (detach para não retropropagar pelo crítico)
            actor_loss = -log_prob * advantages.detach()

            # 5) Perda do crítico = MSE(V(s), td_target)
            value_loss = mse_loss(value_s, td_target)

            # Otimização
            actor_optimizer.zero_grad()
            value_optimizer.zero_grad()
            actor_loss.backward()
            value_loss.backward()
            actor_optimizer.step()
            value_optimizer.step()

            # Atualiza estado
            state = next_state

        writer.add_scalar(f"reward", total_reward, episode)
        writer.add_scalar("actor_loss", actor_loss, episode)
        writer.add_scalar("value_loss", value_loss, episode)
        
    
    print("Treinamento terminado")
    env.close()
    writer.close()

if __name__ == "__main__":
    main()