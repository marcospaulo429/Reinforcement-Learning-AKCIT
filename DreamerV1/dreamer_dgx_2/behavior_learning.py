import torch
import torch.nn.functional as F
import numpy as np
from replay_buffer import ReplayBuffer
from torch.utils.data import TensorDataset

import math

def compute_tanh_entropy(mean, std, num_samples=1000):
    # Gera amostras da distribuição normal base
    u_samples = torch.randn((num_samples, *mean.shape), device=mean.device) * std + mean
    # Calcula a entropia da distribuição normal base para cada dimensão
    base_entropy = 0.5 * torch.log(2 * math.pi * math.e * (std ** 2))
    # Calcula o log do determinante do jacobiano da transformação tanh para cada amostra
    # Adicionamos um pequeno valor (1e-6) para evitar log(0)
    log_det_jacobian = torch.log(1 - torch.tanh(u_samples)**2 + 1e-6)
    # Estima a expectativa do log do jacobiano
    correction = log_det_jacobian.mean(dim=0)
    # A entropia estimada da distribuição transformada é:
    entropy = base_entropy - correction
    # Retorna a média (pode-se adaptar para retornar por dimensão, se necessário)
    return entropy.mean()



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
      - Valores escalares: entropia média, média da média (mean) e média do desvio padrão (std)
        provenientes da última chamada do ator.
    """
    batch_size = latent.size(0)
    latents = []
    rewards = []
    
    for t in range(horizon):
        # Obtém a distribuição do ator para cada latente do batch
        dist, mean, std = actor(latent) 
        action = dist.rsample()   
        
        latent, hidden, _, _ = world_model.transition_model(hidden, latent, action)
        
        r = world_model.reward_model(latent) 
        
        latents.append(latent)
        rewards.append(r)
    
    latents_imag = torch.stack(latents, dim=0)    # (horizon, batch_size, latent_dim)
    rewards_imag = torch.stack(rewards, dim=0)      # (horizon, batch_size, 1)
    
    # Calcula a entropia média, a média dos valores de mean e do std 
    final_mean = mean.mean()               # média de mean
    final_std = std.mean()                 # média de std
    final_entropy = compute_tanh_entropy(mean, std)
    
    return latents_imag, rewards_imag, final_entropy, final_mean, final_std


def behavior_learning( 
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
    Além disso, acumula os valores de entropia, mean e std da política para cada época.
    """
    world_model.train()

    actor_loss_history = []
    value_loss_history = []
    
    # Para acumular os valores da política por época
    epoch_entropy_avg = []
    epoch_mean_avg = []
    epoch_std_avg = []

    for b_ep in range(epochs_behavior):
        epoch_actor_loss = 0.0
        epoch_value_loss = 0.0
        # Acumuladores para os valores de política desta época
        epoch_entropy = 0.0
        epoch_mean = 0.0
        epoch_std = 0.0
        num_batches = 0

        for batch in latent_loader:
            latents, actions, rewards, hiddens = batch
            latents = latents.to(device)
            hiddens = hiddens.to(device)
            batch_size = latents.size(0)

            # Gera rollout imaginado e obtém os valores da política
            latents_imag, rewards_imag, batch_entropy, batch_mean, batch_std = imagine_rollout(
                world_model, actor, latents, hiddens, horizon, gamma
            )
            # latents_imag: (horizon, batch_size, latent_dim)
            # rewards_imag: (horizon, batch_size, 1)

            # Acumula os valores da política para esta batch
            epoch_entropy += batch_entropy
            epoch_mean += batch_mean
            epoch_std += batch_std

            # Cálculo do lambda-return para o value net
            target_values = torch.zeros(horizon, batch_size, 1, device=device)
            
            with torch.no_grad():
                # Bootstrap: valor do último latente
                v_next = value_net(latents_imag[-1])
                target_values[-1] = rewards_imag[-1] + gamma * v_next

                for t in reversed(range(horizon - 1)):
                    v_next = value_net(latents_imag[t + 1])
                    target_values[t] = rewards_imag[t] + gamma * ((1 - lam) * v_next + lam * target_values[t + 1])
            
            actor.train()
            value_net.train()
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

        # Média dos valores de política desta época
        avg_entropy = epoch_entropy / num_batches
        avg_mean = epoch_mean / num_batches
        avg_std = epoch_std / num_batches

        avg_value_loss = epoch_value_loss / num_batches
        avg_actor_loss = epoch_actor_loss / num_batches
        value_loss_history.append(avg_value_loss)
        actor_loss_history.append(avg_actor_loss)
        
        epoch_entropy_avg.append(avg_entropy.item())
        epoch_mean_avg.append(avg_mean.item())
        epoch_std_avg.append(avg_std.item())

    return actor, value_net, actor_loss_history, value_loss_history, epoch_entropy_avg, epoch_mean_avg, epoch_std_avg
