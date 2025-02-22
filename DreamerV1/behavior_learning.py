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
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()  # ações em [-1,1]
        )
    def forward(self, latent):
        return self.net(latent)

class ValueNet(nn.Module):
    def __init__(self, latent_dim):
        super(ValueNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
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
   
def behavior_learning(world_model, actor, value_net, epochs_behavior, train_loader, device, writer,
                      horizon=5, gamma=0.99,
                      value_optimizer=None, actor_optimizer=None,
                      mse_loss=None):

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
            
            # Calcula a perda do value_net ao longo do horizonte
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

        writer.add_scalar("Loss/Value", avg_value_loss, b_ep)
        writer.add_scalar("Loss/Actor", avg_actor_loss, b_ep)
        print(f"Epoch {b_ep+1}/{epochs_behavior} - Actor Loss: {avg_actor_loss:.4f} | Value Loss: {avg_value_loss:.4f}")

    writer.close()
    return actor, value_net

def select_action(actor,state):
        state = torch.FloatTensor(state).unsqueeze(0)
        logits = actor(state)
        action_prob = torch.softmax(logits, dim=-1)
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
    num_episodes =3000
    horizon = 100
    
    reward_history = []

    for episode in range(num_episodes):
        state = env.reset()[0]
        log_probs = []
        rewards = []
        values = []

        for t in range(horizon):
            action, log_prob = select_action(actor,state)
            next_state, reward, done, _, _ = env.step(action)

            log_probs.append(torch.log(log_prob))
            rewards.append(reward)
            values.append(value_net(torch.FloatTensor(state)).squeeze(0))

            state = next_state
            if done:
                break

        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns)

        # Atualizando o ValueNet
        values = torch.stack(values)
        value_loss = mse_loss(values, returns)
        value_optimizer.zero_grad()
        value_loss.backward()
        value_optimizer.step()

        # Atualizando o Actor (Política)
        advantages = returns - values.detach()
        actor_loss = (-torch.stack(log_probs) * advantages).mean()
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        # Salvando histórico de recompensas
        total_reward = sum(rewards)
        reward_history.append(total_reward)

        # Monitoramento
        if (episode+1) % 10 == 0:
            print(f"Episode {episode+1}/{num_episodes}, Total Reward: {total_reward}")

    # Plotando o histórico de recompensas
    plt.plot(reward_history)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Performance do Actor-Critic no CartPole-v1')
    plt.show()

    env.close()
    
if __name__ == "__main__":
    main()