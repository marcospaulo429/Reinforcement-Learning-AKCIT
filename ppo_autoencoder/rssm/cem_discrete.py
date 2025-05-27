import torch
from torch.distributions import Categorical
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cpu")  # Ou use "mps" se desejar

# Configurações do CEM
H = 300            # Horizonte de planejamento
J = 2000           # Número de candidatos por iteração
I = 1000000           # Número de iterações de otimização
K = 100            # Número de melhores candidatos para reajuste

# Criar ambiente
env = gym.make("CartPole-v1")
n_actions = env.action_space.n

# Inicializar probabilidades das ações
probs = (torch.ones(n_actions, device=device) / n_actions).clone()
best_total_reward = -float('inf')
best_actions = None

# TensorBoard writer
writer = SummaryWriter()

# Lista para armazenar recompensas para média acumulada
rewards_history = []

for i in range(I):
    # Gerar sequências de ações candidatas
    action_sequences = torch.stack([
        Categorical(probs).sample((H,))
        for _ in range(J)
    ]).to(device)

    total_rewards = torch.zeros(J, device=device)

    for j in range(J):
        obs, _ = env.reset()
        total_reward = 0.0

        for t in range(H):
            action = action_sequences[j, t].item()
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

            if terminated or truncated:
                break

        total_rewards[j] = total_reward

    # Selecionar as K melhores sequências
    top_rewards, top_indices = torch.topk(total_rewards, K)
    top_actions = action_sequences[top_indices]

    # Atualizar as probabilidades
    probs = torch.zeros(n_actions, device=device)
    for seq in top_actions:
        for a in seq:
            probs[a] += 1
    probs = (probs / (K * H)) + 1e-6
    probs = probs / probs.sum()

    # Melhor recompensa desta iteração
    current_best_reward = top_rewards[0].item()
    rewards_history.append(current_best_reward)
    avg_reward = sum(rewards_history) / len(rewards_history)

    # Log no TensorBoard
    writer.add_scalar("Reward/Best_per_iteration", current_best_reward, i)
    writer.add_scalar("Reward/Average", avg_reward, i)

    # Atualização da melhor sequência global
    if current_best_reward > best_total_reward:
        best_total_reward = current_best_reward
        best_actions = top_actions[0]

    print(f"Iteração {i}: Melhor recompensa desta iteração = {current_best_reward}, Melhor global = {best_total_reward}")

    if best_total_reward >= 500:
        print("Solução ótima encontrada!")
        break

# Executar melhor sequência com renderização
if best_actions is not None:
    obs, _ = env.reset()
    for t in range(H):
        action = best_actions[t].item()
        obs, reward, terminated, truncated, _ = env.step(action)
        env.render()

        if terminated or truncated:
            print(f"Episódio terminado após {t} passos!")
            break

env.close()
writer.close()
