import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent)) 
import torch
import torch.nn.functional as f
import matplotlib.pyplot as plt
from utils.auxiliares import training_device
device = training_device()


def calculate_returns(rewards, discount_factor):
    returns = []
    cumulative_reward = 0
    for r in reversed(rewards):
        cumulative_reward = r + cumulative_reward * discount_factor
        returns.insert(0, cumulative_reward)

    returns = torch.tensor(returns)
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
    advantages = (advantages.detach()).unsqueeze(1)
    policy_ratio = (actions_log_probability_new - actions_log_probability_old).exp()
    surrogate_loss_1 = policy_ratio * advantages
    surrogate_loss_2 = torch.clamp(
            policy_ratio, min=1.0-epsilon, max=1.0+epsilon
            ) * advantages
    surrogate_loss = torch.min(surrogate_loss_1, surrogate_loss_2)
    return surrogate_loss, surrogate_loss_1, surrogate_loss_2, policy_ratio

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

def plot_train_rewards(train_rewards, reward_threshold):
    plt.figure(figsize=(12, 8))
    plt.plot(train_rewards, label='Train Reward')
    plt.xlabel('Episode', fontsize=20)
    plt.ylabel('Training Reward', fontsize=20)
    plt.hlines(reward_threshold, 0, len(train_rewards), color='y')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()

def plot_test_rewards(test_rewards, reward_threshold):
    plt.figure(figsize=(12, 8))
    plt.plot(test_rewards, label='Test Reward')
    plt.xlabel('Episode', fontsize=20)
    plt.ylabel('Testing Reward', fontsize=20)
    plt.hlines(reward_threshold, 0, len(test_rewards), color='y')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()

def plot_losses(policy_losses, value_losses):
    plt.figure(figsize=(12, 8))
    plt.plot(value_losses, label='Value Losses')
    plt.plot(policy_losses, label='Policy Losses')
    plt.xlabel('Episode', fontsize=20)
    plt.ylabel('Loss', fontsize=20)
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()

def load_checkpoint(checkpoint_path, actor, critic, optimizer_actor, optimizer_critic):
    """
    Carrega os pesos salvos dos modelos e otimizadores
    
    Parâmetros:
        checkpoint_path (str): Caminho para o arquivo de checkpoint
        actor: Instância do modelo Actor
        critic: Instância do modelo Critic
        optimizer_actor: Instância do otimizador do Actor
        optimizer_critic: Instância do otimizador do Critic
    
    Retorna:
        Dicionário com informações adicionais do checkpoint (episódio, perdas, recompensas)
    """
    checkpoint = torch.load(checkpoint_path,weights_only=False)
    
    # Carregar pesos dos modelos
    actor.load_state_dict(checkpoint['actor_state_dict'])
    critic.load_state_dict(checkpoint['critic_state_dict'])
    
    # Carregar estados dos otimizadores
    optimizer_actor.load_state_dict(checkpoint['actor_optimizer_state_dict'])
    optimizer_critic.load_state_dict(checkpoint['critic_optimizer_state_dict'])

    print("Checkpoint carregado com sucesso!")
    print(f"Último episódio: {checkpoint['episode']}")
    
    # Retornar outras informações do checkpoint
    return {
        'episode': checkpoint['episode'],
        'policy_loss': checkpoint['policy_loss'],
        'value_loss': checkpoint['value_loss'],
        'reward': checkpoint['reward'],
        'test_reward': checkpoint['test_reward']
    }