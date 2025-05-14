import torch
import numpy as np
from torch.distributions import Normal
import gymnasium as gym

# Configurações do CEM
H = 100          # Horizonte de planejamento
J = 300          # Número de candidatos por iteração
I = 2000           # Número de iterações de otimização
K = 50           # Número de melhores candidatos para reajuste

# Criar ambiente
env = gym.make("MountainCarContinuous-v0")
observation, _ = env.reset()

# Inicializar distribuição das ações (contínuas entre -1 e 1)
mean = torch.zeros(H)
std = torch.ones(H) * 2.0  # Variância maior para explorar mais
best_total_reward = -float('inf')
best_actions = None

for i in range(I):
    # Gerar sequências de ações candidatas
    dist = Normal(mean, std)
    action_sequences = dist.sample((J,))  # Forma (J, H)
    action_sequences = torch.clamp(action_sequences, -1.0, 1.0)  # Limitar entre [-1, 1]
    
    total_rewards = torch.zeros(J)
    
    for j in range(J):
        # Fazer cópia do ambiente para simular sem afetar o estado real
        temp_env = gym.make("MountainCarContinuous-v0")
        temp_env.reset()
        temp_env.unwrapped.state = env.unwrapped.state  # Copiar estado atual
        
        total_reward = 0.0
        
        for t in range(H):
            action = action_sequences[j, t].item()
            obs, reward, terminated, truncated, _ = temp_env.step([action])
            total_reward += reward
            
            if terminated or truncated:
                break
                
        total_rewards[j] = total_reward
        temp_env.close()
    
    # Selecionar as K melhores sequências
    top_rewards, top_indices = torch.topk(total_rewards, K)
    top_actions = action_sequences[top_indices]
    
    # Atualizar a distribuição
    mean = top_actions.mean(dim=0)
    std = top_actions.std(dim=0, unbiased=True) + 1e-6  # Evitar std zero
    
    # Manter registro da melhor sequência encontrada
    current_best_reward = top_rewards[0].item()
    print(current_best_reward)
    if current_best_reward > best_total_reward:
        best_total_reward = current_best_reward
        best_actions = top_actions[0]

    if current_best_reward > 0:
        print(current_best_reward)
        break

# Executar a melhor sequência encontrada no ambiente real
if best_actions is not None:
    for t in range(H):
        action = best_actions[t].item()
        observation, reward, terminated, truncated, _ = env.step([action])
        
        if terminated or truncated:
            break

env.close()