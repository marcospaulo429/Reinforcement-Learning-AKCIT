import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from replay_buffer import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset

class Actor(nn.Module):
    def __init__(self, latent_dim, action_dim):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()  # ações em [-1,1]
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
    hidden = initial_hidden
    latent = initial_latent

    for t in range(horizon):
        action = actor(latent)
        # Avança o estado latente
        latent, hidden, mean, std = world_model.transition_model(hidden, latent, action)
        r = world_model.reward_model(latent)
        latents.append(latent)
        rewards.append(r)
    
    returns = []
    ret = torch.zeros_like(rewards[-1])
    for r in reversed(rewards):
        ret = r + gamma * ret
        returns.insert(0, ret)
    
    latents = torch.stack(latents, dim=0)   # (H, B, lat_dim)
    rewards = torch.stack(rewards, dim=0)   # (H, B, 1) ou algo assim
    returns = torch.stack(returns, dim=0)   # (H, B, 1)
    return latents, rewards, returns

   
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

            # Gera rollout imaginado no espaço latente a partir de s_t (latents)
            # latents_imag.shape = (horizon, batch_size, latent_dim)
            # rewards_imag.shape = (horizon, batch_size, 1)
            # [Podemos ignorar 'done' se seu world_model não modela terminação]
            latents_imag, rewards_imag, _ = imagine_rollout(
                world_model, actor, latents, hidden_init, horizon, gamma
            )

            # -------------------------------------------------------------
            # Cálculo do λ-return: para cada t no rollout, calculamos o alvo
            # para o value_net (que chamaremos de target_value[t]).
            # No Dreamer, costuma-se fazer:  V_λ(s_t) = (1-λ) * sum(λ^k * n-step) + ...
            # Aqui faremos algo mais direto: para t = H-1..0 (backward):
            #   G[t] = r[t] + gamma * [(1-lambda) * V(s_{t+1}) + lambda * G[t+1]]
            #   target_value[t] = G[t]
            # (onde V(s_{t+1}) é a predição do value_net no próximo estado imaginado)
            # -------------------------------------------------------------

            # Prepara tensores para armazenar os targets
            # shape: (horizon, batch_size, 1)
            target_values = torch.zeros(horizon, batch_size, 1, device=device)

            # Para o último passo do rollout, bootstrap do value_net
            # (poderia ser zero se quiser horizon "fixo", mas Dreamer normalmente faz bootstrap)
            v_next = value_net(latents_imag[-1])  # V(s_{H-1})
            target_values[-1] = rewards_imag[-1] + gamma * v_next * (1 - lam)  \
                                + gamma * lam * v_next  # ou simplifique se preferir

            # Retropropaga do penúltimo passo até o primeiro
            for t in reversed(range(horizon - 1)):
                v_next = value_net(latents_imag[t + 1])  # V(s_{t+1})
                target_values[t] = rewards_imag[t] + gamma * (
                    (1 - lam) * v_next + lam * target_values[t + 1]
                )

            # -------------------------------------------------------------
            # LOSS DO VALUE NET:
            # Fazemos MSE entre V(s_t) e target_values[t] para cada t
            # -------------------------------------------------------------
            value_loss = 0.0
            for t in range(horizon):
                v_pred = value_net(latents_imag[t])
                value_loss += mse_loss(v_pred, target_values[t])
            value_loss /= horizon

            # Otimiza o value_net
            value_optimizer.zero_grad()
            value_loss.backward(retain_graph=True)
            value_optimizer.step()

            # -------------------------------------------------------------
            # LOSS DO ACTOR:
            # O ator é otimizado para maximizar a soma dos valores previstos
            # ou, equivalentemente, minimizar a negativa dessa soma.
            # (No Dreamer, também se usa "policy entropy" e outras regularizações.)
            # Aqui: actor_loss = - mean( V(s_t) ) ao longo do rollout.
            # -------------------------------------------------------------
            # Podemos usar a média dos V(s_t) ou a média dos targets G[t].
            # O Dreamer normalmente faz backprop via V(s_t) (que depende das ações
            # escolhidas no rollout) e *não* desconecta o grafo, permitindo
            # gradientes fluírem pelo world_model e ator.
            # Ex: actor_loss = - (1/horizon) * sum_{t=0..H-1} V(s_t)
            # ou, se quiser, use 'target_values[t]' sem grad (mas isso não backprop
            # pelo modelo dinâmico).
            # Abaixo, uso V(s_t) para permitir gradiente analítico:
            actor_loss = 0.0
            for t in range(horizon):
                actor_loss += -value_net(latents_imag[t]).mean()
            actor_loss /= horizon

            actor_optimizer.zero_grad()
            actor_loss.backward()
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