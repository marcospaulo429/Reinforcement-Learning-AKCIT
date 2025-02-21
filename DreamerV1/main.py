# main.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from dm_control import suite
from dm_control.suite.wrappers import pixels

from replay_buffer import ReplayBuffer
from world_model import DreamerWorldModel, converter_cinza, get_data_loaders_from_replay_buffer, ver_reconstrucoes, collect_replay_buffer, train_world_model
from behavior_learning import Actor, ValueNet
from torch.utils.tensorboard import SummaryWriter

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
                      horizon=5, gamma=0.99,
                      value_optimizer=None, actor_optimizer=None,
                      mse_loss=None, repositorio=None):
    writer = SummaryWriter(f"{repositorio}")

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

            actor_loss = -rewards.mean()
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



def environment_interaction(env, actor, world_model, replay_buffer, device, 
                            hidden_dim, steps=500, exploration_noise=0.1):
    time_step = env.reset()
    done = False
    obs_atual = converter_cinza(time_step.observation['pixels'])
    obs_atual = obs_atual.astype(np.float32) / 127.5 - 1.0

    episode_data = []
    steps_done = 0

    while steps_done < steps:
        # Converte a observação para tensor com shape (1, 1, 84, 84)
        obs_tensor = torch.tensor(obs_atual).view(1, 1, 84, 84).to(device)
        conv_out = world_model.autoencoder.encoder_conv(obs_tensor)  # (1, 64, 21, 21)
        conv_out = conv_out.view(conv_out.size(0), -1)               # (1, 28224)
        latent = world_model.autoencoder.encoder_fc(conv_out)        # (1, latent_dim)
        
        action_tensor = actor(latent)
        action_np = action_tensor.detach().cpu().numpy()[0]
        
        # TODO Estratégia de exploração

        time_step = env.step(action_np)
        done = time_step.last()
        reward = time_step.reward if time_step.reward is not None else 0.0

        obs_prox = converter_cinza(time_step.observation['pixels'])
        obs_prox = obs_prox.astype(np.float32) / 127.5 - 1.0

        step_data = {
            "obs": obs_atual,
            "action": action_np,
            "reward": reward,
            "next_obs": obs_prox,
            "done": done
        }
        episode_data.append(step_data)
        obs_atual = obs_prox
        steps_done += 1

        if done:
            replay_buffer.add_episode(episode_data)
            episode_data = []
            time_step = env.reset()
            done = False
            obs_atual = converter_cinza(time_step.observation['pixels'])
            obs_atual = obs_atual.astype(np.float32) / 127.5 - 1.0

def evaluate_policy(env, actor, world_model, device, num_episodes=5): 
    rewards_ep = []
    for _ in range(num_episodes):
        time_step = env.reset()
        done = False
        ep_reward = 0.0

        obs_atual = converter_cinza(time_step.observation['pixels'])
        obs_atual = obs_atual.astype(np.float32) / 127.5 - 1.0

        while not done:
            obs_tensor = torch.tensor(obs_atual).view(1, 1, 84, 84).to(device)
            conv_out = world_model.autoencoder.encoder_conv(obs_tensor)  # (1, 64, 21, 21)
            conv_out = conv_out.view(conv_out.size(0), -1)               # (1, 28224)
            latent = world_model.autoencoder.encoder_fc(conv_out)        # (1, latent_dim)
            
            action_tensor = actor(latent)
            action_np = action_tensor.detach().cpu().numpy()[0]
            
            time_step = env.step(action_np)
            done = time_step.last()
            reward = time_step.reward if time_step.reward is not None else 0.0
            ep_reward += reward
            
            obs_atual = converter_cinza(time_step.observation['pixels'])
            obs_atual = obs_atual.astype(np.float32) / 127.5 - 1.0

        rewards_ep.append(ep_reward)
    return rewards_ep

def main():
    HEIGHT = 84
    WIDTH = 84
    hidden_dim = 256
    input_size = HEIGHT * WIDTH
    latent_dim = 256
    batch_size = 32
    S = 5
    repositorio = "dreamer/model_1"
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    print("Usando device:", device)
    
    env = suite.load(domain_name="cartpole", task_name="balance")
    env = pixels.Wrapper(env, pixels_only=True,
                         render_kwargs={'height': HEIGHT, 'width': WIDTH, 'camera_id': 0})
    
    replay_buffer = ReplayBuffer()
    replay_buffer = collect_replay_buffer(env, S, replay_buffer)
    
    action_dim = env.action_spec().shape[0]
    
    world_model = DreamerWorldModel(input_size, latent_dim, action_dim, hidden_dim).to(device)
    world_model.load_state_dict(torch.load("world_model/model_3/world_model_weights.pth"))
    wm_optimizer = optim.Adam(world_model.parameters(), lr=0.001)
    mse_loss = nn.MSELoss()
    
    actor = Actor(latent_dim, action_dim).to(device)
    value_net = ValueNet(latent_dim).to(device)
    actor_optimizer = optim.Adam(actor.parameters(), lr=1e-3)
    value_optimizer = optim.Adam(value_net.parameters(), lr=1e-3)
    
    train_loader, test_loader = get_data_loaders_from_replay_buffer(replay_buffer, batch_size=batch_size, HEIGHT=HEIGHT, WIDTH=WIDTH)
    
    writer = SummaryWriter(f"{repositorio}")
    
    rewards_history = []
    #num_iterations = int(input("Coloque o número de interações: "))
    num_iterations = 10
    iteration = 0
    
    while num_iterations > 0:
        for it in range(num_iterations):
            iteration += 1
            print(f"\n Iteração {it+1}/{num_iterations} (Total Iterações: {iteration})")
            
            epochs_wm = 5
            train_world_model(epochs_wm, world_model, train_loader, test_loader, device, hidden_dim, mse_loss, wm_optimizer, repositorio)
            
            epochs_behavior = 5
            actor, value_net = behavior_learning(world_model, actor, value_net, epochs_behavior, train_loader, device,
                                                  horizon=5, gamma=0.99,
                                                  value_optimizer=value_optimizer, actor_optimizer=actor_optimizer,
                                                  mse_loss=mse_loss, repositorio=repositorio)
            
            environment_interaction(env, actor, world_model, replay_buffer, device,
                                    hidden_dim=hidden_dim, steps=300, exploration_noise=0.1)
            train_loader, test_loader = get_data_loaders_from_replay_buffer(replay_buffer, batch_size=32, HEIGHT=HEIGHT, WIDTH=WIDTH)
            
            ep_rewards = evaluate_policy(env, actor, world_model, device, num_episodes=3)
            avg_rew = np.mean(ep_rewards)
            rewards_history.append(avg_rew)
            print(f"  Recompensa média (avaliação) = {avg_rew:.2f}")
            writer.add_scalar("Reward/Average", avg_rew, iteration)
            
            for batch in test_loader:
                obs, _, _, _ = next(iter(train_loader))

                if obs.dim() == 2 and obs.size(1) == 84*84:
                    obs = obs.view(obs.size(0), 1, 84, 84)

                obs = obs.to(device)
                conv_out = world_model.autoencoder.encoder_conv(obs)  # Saída: (B, 64, 21, 21)
                conv_out = conv_out.view(conv_out.size(0), -1)           # (B, 28224)
                latent = world_model.autoencoder.encoder_fc(conv_out)    # (B, latent_dim)

                writer.add_histogram("Latent/Distribution", latent, iteration)
                break  
        
        print("\nTreinamento finalizado para essa fase!")
        ver_reconstrucoes(world_model, test_loader, device, input_size, num_samples=8,
                          action_dim=action_dim, hidden_dim=hidden_dim, HEIGHT=HEIGHT, WIDTH=WIDTH)
        
        num_iterations = int(input("Coloque o número de interações (0 para finalizar): "))
    
    print("Treinamento finalizado com sucesso, veja as métricas no TensorBoard.")
    writer.close()

if __name__ == "__main__":
    main()
