import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from replay_buffer import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset
from actor_critic import ActionModel, ValueNet
from auxiliares import training_device

def extract_latent_sequences(world_model, replay_buffer, device):
    """
    Percorre o replay_buffer original (com observações/imagens) e gera
    um novo buffer com sequências de latentes, usando o world_model treinado.
    """
    world_model.eval()
    
    latent_buffer = ReplayBuffer() 

    with torch.no_grad():
        for episode in replay_buffer.buffer:
            latent_episode = []
            
            hidden_dim = world_model.transition_model.gru.hidden_size
            # Inicializa o estado oculto para um batch de tamanho 1
            hidden = torch.zeros(1, hidden_dim, device=device)

            for step in episode:
                # Converte obs/action para tensores
                obs_t = torch.tensor(step["obs"], dtype=torch.float32, device=device)
                # Se obs_t for (84*84,), reshape para (1, 1, 84, 84)
                obs_t = obs_t.view(1, 1, 84, 84)
                
                action_t = torch.tensor(step["action"], dtype=torch.float32, device=device).unsqueeze(0)
                
                # Forward pass para obter o estado latente da observação
                latent_next, hidden, _, _, _, _ = world_model(obs_t, action_t, hidden)
                
                # Converte latent_next para CPU/Numpy para armazenar fora do PyTorch
                latent_next_np = latent_next.squeeze(0).cpu().numpy()

                # Armazena também o estado oculto
                hidden_np = hidden.squeeze(0).cpu().numpy()

                # Monta um dicionário para armazenar no novo buffer
                latent_step = {
                    "latent": latent_next_np,
                    "action": step["action"],
                    "reward": step["reward"],
                    "hidden": hidden_np,
                }
                latent_episode.append(latent_step)
            
            # Ao final do episódio, adiciona a sequência latente como um novo episódio
            latent_buffer.add_episode(latent_episode)
    
    return latent_buffer


def create_latent_dataset(latent_buffer):
    """
    Converte cada episódio em arrays e concatena para formar um TensorDataset
    contendo latents, ações, recompensas e hiddens.
    """
    latents = []
    actions = []
    rewards = []
    hiddens = []

    for ep in latent_buffer.buffer:
        for step in ep:
            latents.append(step["latent"])
            actions.append(step["action"])
            rewards.append(step["reward"])
            hiddens.append(step["hidden"])

    latents = np.array(latents, dtype=np.float32)
    actions = np.array(actions, dtype=np.float32)
    rewards = np.array(rewards, dtype=np.float32)
    hiddens = np.array(hiddens, dtype=np.float32)
    
    # Converte rewards para shape (N, 1)
    rewards = np.expand_dims(rewards, axis=-1)
    
    dataset = TensorDataset(
        torch.from_numpy(latents),
        torch.from_numpy(actions),
        torch.from_numpy(rewards),
        torch.from_numpy(hiddens)
    )
    return dataset


def imagine_rollout(world_model, actor, latent, hidden, horizon=5, gamma=0.99):
    """
    Realiza o rollout imaginado de forma vetorizada.
    latent: tensor de shape (batch_size, latent_dim)
    hidden: tensor de shape (batch_size, hidden_dim)
    
    Retorna:
      - latents_imag: tensor de shape (horizon, batch_size, latent_dim)
      - rewards_imag: tensor de shape (horizon, batch_size, 1)
    """
    batch_size = latent.size(0)
    latents = []
    rewards = []

    for t in range(horizon):
        # Obtém a distribuição do ator para cada latente do batch
        dist, _, _ = actor(latent) 
        action = dist.rsample()   
        
        latent, hidden, _, _ = world_model.transition_model(hidden, latent, action)
        
        r = world_model.reward_model(latent) 
        
        latents.append(latent)
        rewards.append(r)
    
    latents_imag = torch.stack(latents, dim=0)    # (horizon, batch_size, latent_dim)
    rewards_imag = torch.stack(rewards, dim=0)      # (horizon, batch_size, 1)
    
    return latents_imag, rewards_imag


def behavior_learning( #TODO: ver se funcoes de valor e modelo de actor-critic esta certo
    world_model, actor, value_net,
    latent_loader,
    device,
    horizon=15,         
    gamma=0.99,
    lam=0.95,          
    value_optimizer=None,
    actor_optimizer=None,
    mse_loss=None,
    epochs_behavior=10
):
    """
    Treina o ator e o value net utilizando rollouts imaginados a partir dos latentes.
    """
    actor_loss_history = []
    value_loss_history = []

    for b_ep in range(epochs_behavior):
        epoch_actor_loss = 0.0
        epoch_value_loss = 0.0
        num_batches = 0

        for batch in latent_loader:
            latents, actions, rewards, hiddens = batch
            latents = latents.to(device)
            hiddens = hiddens.to(device)
            batch_size = latents.size(0)

            # Gera rollout imaginado de forma vetorizada a partir dos estados latentes atuais
            latents_imag, rewards_imag = imagine_rollout(
                world_model, actor, latents, hiddens, horizon, gamma
            )
            # latents_imag: (horizon, batch_size, latent_dim)
            # rewards_imag: (horizon, batch_size, 1)

            # Cálculo do lambda-return para o value net
            target_values = torch.zeros(horizon, batch_size, 1, device=device)
            with torch.no_grad():
                # Bootstrap: valor do último latente
                v_next = value_net(latents_imag[-1])
                target_values[-1] = rewards_imag[-1] + gamma * v_next

                for t in reversed(range(horizon - 1)):
                    v_next = value_net(latents_imag[t + 1])
                    target_values[t] = rewards_imag[t] + gamma * ((1 - lam) * v_next + lam * target_values[t + 1])
            
            # Cálculo da loss do value net
            value_loss = 0.0
            for t in range(horizon):
                v_pred = value_net(latents_imag[t])
                value_loss += mse_loss(v_pred, target_values[t])
            value_loss /= horizon

            value_optimizer.zero_grad()
            value_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(value_net.parameters(), max_norm=100)
            value_optimizer.step()

            # Cálculo da loss do ator
            actor_loss = 0.0
            for t in range(horizon):
                # Atualiza o ator para maximizar o valor predito
                actor_loss += -value_net(latents_imag[t]).mean()
            actor_loss /= horizon

            actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=100)
            actor_optimizer.step()

            epoch_value_loss += value_loss.item()
            epoch_actor_loss += actor_loss.item()
            num_batches += 1

        avg_value_loss = epoch_value_loss / num_batches
        avg_actor_loss = epoch_actor_loss / num_batches
        value_loss_history.append(avg_value_loss)
        actor_loss_history.append(avg_actor_loss)

        print(f"[Behavior] Epoch {b_ep+1}/{epochs_behavior} | "
              f"Value Loss: {avg_value_loss:.4f} | Actor Loss: {avg_actor_loss:.4f}")

    return actor, value_net, actor_loss_history, value_loss_history
