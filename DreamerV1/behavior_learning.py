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
    
    latent_buffer = ReplayBuffer()  # Buffer para guardar (s_lat, a, r, s_lat_next, ...)

    with torch.no_grad():
        for episode in replay_buffer.buffer:
            latent_episode = []
            
            hidden_dim = world_model.transition_model.gru.hidden_size
            # Inicializa hidden como 2D: (batch, hidden_dim), onde batch=1
            hidden = torch.zeros(1, hidden_dim, device=device)

            for step in episode:
                # Converte obs/action para tensores
                obs_t = torch.tensor(step["obs"], dtype=torch.float32, device=device)
                # Se obs_t for (84*84,), reshape para (1, 1, 84, 84) se necessário
                obs_t = obs_t.view(1, 1, 84, 84)
                
                action_t = torch.tensor(step["action"], dtype=torch.float32, device=device).unsqueeze(0)
                
                # Forward pass para obter o estado latente da observação
                latent_next, hidden, _, _, _, _ = world_model(obs_t, action_t, hidden)
                
                # Converte latent_next para CPU/Numpy para armazenar fora do PyTorch
                latent_next_np = latent_next.squeeze(0).cpu().numpy()

                # Monta um dicionário para armazenar no novo buffer
                latent_step = {
                    "latent": latent_next_np,
                    "action": step["action"],
                    "reward": step["reward"],
                }
                latent_episode.append(latent_step)
            
            # Ao final do episódio, adiciona a sequência latente como um novo episódio
            latent_buffer.add_episode(latent_episode)
    
    return latent_buffer


def create_latent_dataset(latent_buffer):
    # Converte cada episódio em arrays e concatena
    latents = []
    actions = []
    rewards = []
    latents_next = []

    for ep in latent_buffer.buffer:
        for step in ep:
            latents.append(step["latent"])
            actions.append(step["action"])
            rewards.append(step["reward"])
            # se estiver armazenando latent_next
            if "latent_next" in step:
                latents_next.append(step["latent_next"])
            else:
                latents_next.append(np.zeros_like(step["latent"]))  # dummy se não tiver

    latents = np.array(latents, dtype=np.float32)
    actions = np.array(actions, dtype=np.float32)
    rewards = np.array(rewards, dtype=np.float32)
    latents_next = np.array(latents_next, dtype=np.float32)
    
    # Monta um TensorDataset
    dataset = TensorDataset(
        torch.from_numpy(latents),
        torch.from_numpy(actions),
        torch.from_numpy(rewards),
        torch.from_numpy(latents_next)
    )
    return dataset


def imagine_rollout(world_model, actor, initial_latent, initial_hidden, horizon=5, gamma=0.99):
    latents = []
    rewards = []
    log_probs = []
    entropies = []
    hidden = initial_hidden
    latent = initial_latent

    for t in range(horizon):
        # Obter a distribuição do ator para o estado latente atual
        dist, mean, std = actor(latent)
        # Amostrar ação com reparametrização
        action = dist.rsample()
        # Calcular log probability e entropia
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        entropy = dist.base_dist.entropy().sum(dim=-1, keepdim=True)
        log_probs.append(log_prob)
        entropies.append(entropy)
        
        # Avança o estado latente usando o modelo de transição
        latent, hidden, trans_mean, trans_std = world_model.transition_model(hidden, latent, action)
        # Previsão da recompensa
        r = world_model.reward_model(latent)
        latents.append(latent)
        rewards.append(r)
    
    # Cálculo dos retornos (neste exemplo, sem bootstrapping com V(s_H))
    returns = []
    ret = torch.zeros_like(rewards[-1])
    for r in reversed(rewards):
        ret = r + gamma * ret
        returns.insert(0, ret)
    
    latents = torch.stack(latents, dim=0)   # [horizon, batch, latent_dim]
    rewards = torch.stack(rewards, dim=0)     # [horizon, batch, 1]
    returns = torch.stack(returns, dim=0)     # [horizon, batch, 1]
    log_probs = torch.stack(log_probs, dim=0) # [horizon, batch, 1]
    entropies = torch.stack(entropies, dim=0) # [horizon, batch, 1]
    
    return latents, rewards, returns, log_probs, entropies


   
def behavior_learning(
    world_model, actor, value_net,
    latent_loader,
    device,
    horizon=15,         # Tamanho do rollout imaginado
    gamma=0.99,
    lam=0.95,           # Fator lambda para lambda-return
    value_optimizer=None,
    actor_optimizer=None,
    mse_loss=None,
    epochs_behavior=10
):
    actor_loss_history = []
    value_loss_history = []

    for b_ep in range(epochs_behavior):
        epoch_actor_loss = 0.0
        epoch_value_loss = 0.0
        num_batches = 0

        for batch in latent_loader:
            latents, actions, rewards, latents_next = batch
            latents = latents.to(device)
            batch_size = latents.size(0)

            # Inicializa estado oculto (GRU) para o rollout
            hidden_dim = world_model.transition_model.gru.hidden_size
            hidden_init = torch.zeros(batch_size, hidden_dim, device=device)

            # Gera rollout imaginado a partir dos estados latentes atuais
            latents_imag, rewards_imag, target_values, log_probs, entropies = imagine_rollout(
                world_model, actor, latents, hidden_init, horizon, gamma
            )
            
            # Calcula as vantagens para cada timestep: target - V(s)
            advantages = []
            for t in range(horizon):
                v_pred = value_net(latents_imag[t])
                advantage = target_values[t] - v_pred
                advantages.append(advantage)
            advantages = torch.stack(advantages, dim=0)  # [horizon, batch, 1]

            # Normaliza as vantagens para cada timestep separadamente sobre o batch
            advantages_norm = []
            for t in range(horizon):
                adv = advantages[t]
                adv_mean = adv.mean()
                adv_std = adv.std() + 1e-8
                advantages_norm.append((adv - adv_mean) / adv_std)
            advantages_norm = torch.stack(advantages_norm, dim=0)  # [horizon, batch, 1]

            # Cálculo do λ-return para o value net
            target_values = torch.zeros(horizon, batch_size, 1, device=device)
            v_next = value_net(latents_imag[-1])
            target_values[-1] = rewards_imag[-1] + gamma * v_next * (1 - lam) + gamma * lam * v_next
            
            for t in reversed(range(horizon - 1)):
                v_next = value_net(latents_imag[t + 1])
                target_values[t] = rewards_imag[t] + gamma * ((1 - lam) * v_next + lam * target_values[t + 1])
            
            # LOSS DO VALUE NET
            value_loss = 0.0
            for t in range(horizon):
                v_pred = value_net(latents_imag[t])
                value_loss += mse_loss(v_pred, target_values[t])
            value_loss /= horizon

            value_optimizer.zero_grad()
            torch.nn.utils.clip_grad_norm_(value_net.parameters(), max_norm=10.0)
            value_optimizer.step()

            # Coeficiente de entropia (hiperparâmetro)
            entropy_coef = 0.01

            # Calcula a loss do ator com base no log_prob ponderado pela vantagem normalizada e
            # adiciona a regularização de entropia
            actor_loss = 0.0
            for t in range(horizon):
                # Negativo para maximizar a vantagem
                loss_t = - (log_probs[t] * advantages_norm[t]).mean() - entropy_coef * entropies[t].mean()
                actor_loss += loss_t
            actor_loss /= horizon

            actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=10.0)
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



def select_action(actor, state):
    """
    Dado o estado (vetorial), retorna a ação contínua e seu log-prob.
    """
    state_t = torch.FloatTensor(state).unsqueeze(0)  # [1, state_dim]
    dist, mean, std = actor(state_t)
    # Usamos rsample para permitir backprop (reparametrização)
    action_t = dist.rsample()  
    log_prob = dist.log_prob(action_t).sum(dim=-1)
    action = action_t.detach().cpu().numpy()[0]
    return action, log_prob



def main():
    device = training_device()
    env = gym.make('Pendulum-v1')
    max_episode_steps = 300
    obs_dim = env.observation_space.shape[0]   # geralmente 3 para Pendulum
    action_dim = env.action_space.shape[0]       # 1 dimensão
    latent_dim = obs_dim  # Aqui, tratamos o estado observado como latente

    actor = ActionModel(latent_dim, action_dim)
    value_net = ValueNet(latent_dim)

    actor_optimizer = optim.Adam(actor.parameters(), lr=1e-5)
    value_optimizer = optim.Adam(value_net.parameters(), lr=1e-5)
    mse_loss = nn.MSELoss()
    gamma = 0.99
    num_episodes = 1000
    repo = "behavior/model_continuous_1"
    writer = SummaryWriter(repo)
    print("Começando episodios")

    for episode in range(num_episodes):
        print(episode)
        state, _ = env.reset()
        done = False
        total_reward = 0
        step_count = 0

        while not (done or (step_count==max_episode_steps)):
            
            step_count += 1

            # Seleciona ação a partir do ator (distribuição contínua)
            action, log_prob = select_action(actor, state)
            next_state, reward, done, truncated, _ = env.step(action)
            total_reward += reward

            # Converte estados para tensores
            state_t = torch.FloatTensor(state).unsqueeze(0)
            next_state_t = torch.FloatTensor(next_state).unsqueeze(0)

            # Calcula V(s) e V(s')
            value_s = value_net(state_t)
            with torch.no_grad():
                value_next = value_net(next_state_t)
                if done or truncated:
                    value_next = torch.zeros_like(value_next)
            
            # TD target: r + γ V(s')
            td_target = torch.FloatTensor([[reward]]) + gamma * value_next

            # Vantagem: TD error
            advantage = td_target - value_s

            # Loss do ator: maximiza valor (minimiza -log_prob * advantage)
            actor_loss = -log_prob * advantage.detach()

            # Loss do crítico: MSE entre V(s) e TD target
            value_loss = mse_loss(value_s, td_target)

            # Otimização
            actor_optimizer.zero_grad()
            value_optimizer.zero_grad()
            actor_loss.backward()
            value_loss.backward()
            actor_optimizer.step()
            value_optimizer.step()

            state = next_state

        writer.add_scalar("reward", total_reward, episode)
        writer.add_scalar("actor_loss", actor_loss.item(), episode)
        writer.add_scalar("value_loss", value_loss.item(), episode)
        print(f"Episode {episode+1}/{num_episodes} | Reward: {total_reward:.2f}")

    print("Treinamento terminado")
    env.close()
    writer.close()

if __name__ == "__main__":
    main()