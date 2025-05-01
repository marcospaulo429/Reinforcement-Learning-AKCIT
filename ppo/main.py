import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent)) 

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
from dm_control import suite
from dm_control.suite.wrappers import pixels
from utils.auxiliares import training_device, load_config
device = training_device()
import wandb
from models.autoencoder import VAE
from models.actor_critic import Actor, Critic
from utils.ppo_functions import (
    load_checkpoint
)
from utils.train import evaluate, forward_pass, update_policy

torch.autograd.set_detect_anomaly(True)

def create_agent(HIDDEN_DIMENSIONS, DROPOUT, INPUT_FEATURES, ACTOR_OUTPUT_FEATURES,NUM_LAYERS):
    CRITIC_OUTPUT_FEATURES = 1

    actor = Actor(
            INPUT_FEATURES, HIDDEN_DIMENSIONS, ACTOR_OUTPUT_FEATURES,num_layers=NUM_LAYERS).to(device)
    critic = Critic(
            INPUT_FEATURES, HIDDEN_DIMENSIONS, CRITIC_OUTPUT_FEATURES, DROPOUT).to(device)
    return actor, critic



# 1. Defina os mesmos parâmetros usados no treinamento
latent_dim = 128  # Substitua pelo valor usado originalmente
in_channels = 1   # Substitua pelo valor usado originalmente
hidden_units = 64 # Substitua pelo valor usado originalmente

# 2. Instancie o modelo e otimizador (igual ao do treinamento)
model = VAE(latent_dim=latent_dim, in_channels=in_channels, hidden_units=hidden_units).to(device)
optimizer_encoder = torch.optim.Adam(model.parameters(), lr=1e-3)  # Ou o otimizador que você usou

# 3. Caminho para o checkpoint
#checkpoint_dir = "models/checkpoint_dgx_treino3-lat128_lr0.0001_h128_epoch570.pt" #TODO
#epoch_to_load = 570                           # Epoch que deseja carregar

#model = model.load_checkpoint(checkpoint_dir,epoch_to_load, model, optimizer_encoder)
encoder = model.encoder


env_train = suite.load(domain_name="cartpole", task_name="swingup")
env_train = pixels.Wrapper(env_train, pixels_only=True,
                    render_kwargs={'height': 84, 'width': 84, 'camera_id': 0})
# Teste com imagens do ambiente
env_test = suite.load(domain_name="cartpole", task_name="swingup")
env_test = pixels.Wrapper(env_train, pixels_only=True,
                    render_kwargs={'height': 84, 'width': 84, 'camera_id': 0})


def random_search_hyperparameters():
    ppo_steps = np.random.randint(7, 12)  
    
    # Amostra NUM_LAYERS (inteiro uniforme)
    num_layers = np.random.randint(1, 4) 
    
    # Amostra HIDDEN_DIMENSIONS (inteiro em escala logarítmica)
    hidden_dim =  np.random.randint(80, 100) 
    
    # Amostra LEARNING_RATE_ACTOR e LEARNING_RATE_CRITIC (log-uniform)
    # Correção: Geração correta de learning rates log-uniform entre 1e-5 e 1
    log_min, log_max = np.log10(1e-4), np.log10(1e-3)#TODO
    log_lr = np.random.uniform(log_min, log_max)
    lr = 10**log_lr
    
    return {
        'PPO_STEPS': ppo_steps,
        'NUM_LAYERS': num_layers,
        'HIDDEN_DIMENSIONS': hidden_dim,
        'LEARNING_RATE_ACTOR': lr,
        'LEARNING_RATE_CRITIC': lr
    }

def get_batch_size(size, min_batch=256, max_batch=520):
    """
    Escolhe um tamanho de batch entre min_batch e max_batch que divide igualmente o tamanho total.
    
    Args:
        size (int): Tamanho total do dataset
        min_batch (int): Tamanho mínimo do batch (default: 16)
        max_batch (int): Tamanho máximo do batch (default: 64)
    
    Returns:
        int: Tamanho do batch que divide o dataset igualmente
    """
    # Encontra todos os divisores de size entre min_batch e max_batch
    possible_batches = [b for b in range(min_batch, max_batch+1) if size % b == 0]
    
    if not possible_batches:
        # Se não encontrar divisores perfeitos, usa o maior batch possível que não exceda max_batch
        possible_batches = [b for b in range(min_batch, max_batch+1) if b <= size]
        if not possible_batches:
            return size  # Caso o dataset seja menor que min_batch
    
    # Retorna o maior batch possível que divide igualmente
    return max(possible_batches)

def run_ppo(): 
    
    config = load_config()
    
    device = training_device()
    use_wandb = config['training']['use_wandb']
    use_checkpoint = config['training']['use_checkpoint']
    LATENT_DIM = config['model']['latent_dim']
    IN_CHANNELS = config['model']['in_channels']
    HIDDEN_UNITS = config['model']['hidden_units']
    INPUT_FEATURES = config['model']['input_features']
    ACTOR_OUTPUT_FEATURES = env_test.action_spec().shape[0]
    MAX_EPISODES = config['training']['max_episodes']
    DISCOUNT_FACTOR = config['training']['discount_factor']
    REWARD_THRESHOLD = config['training']['reward_threshold']
    PRINT_INTERVAL = config['training']['print_interval']
    PPO_STEPS = config['training']['ppo_steps']
    N_TRIALS = config['training']['n_trials']
    EPSILON = config['training']['epsilon']
    ENTROPY_COEFFICIENT = config['training']['entropy_coefficient']
    NUM_LAYERS = config['model']['num_layers']
    HIDDEN_DIMENSIONS = config['model']['hidden_dimensions']
    DROPOUT = config['model']['dropout']
    LEARNING_RATE_ACTOR = config['optimizer']['learning_rate_actor']
    LEARNING_RATE_CRITIC = config['optimizer']['learning_rate_critic']
    QUANTITY_RANDOM_SEARCH = config['training']['quantity_random_search']
    NAME_EXPERIMENT = config['training']['name_experiment']
    save_model = config['training']['save_model']


    for i in range(QUANTITY_RANDOM_SEARCH): #fazendo 
        # Executa a função para gerar os hiperparâmetros
        hyperparams = random_search_hyperparameters()

        # Extrai cada valor do dicionário para variáveis separadas
        PPO_STEPS = hyperparams['PPO_STEPS']
        NUM_LAYERS = hyperparams['NUM_LAYERS']
        HIDDEN_DIMENSIONS = hyperparams['HIDDEN_DIMENSIONS']
        LEARNING_RATE_ACTOR = hyperparams['LEARNING_RATE_ACTOR']
        LEARNING_RATE_CRITIC = hyperparams['LEARNING_RATE_CRITIC']

        best_reward = 0
        name_wandb = f"{NAME_EXPERIMENT}_{LEARNING_RATE_ACTOR}_hidden{HIDDEN_DIMENSIONS}"
        checkpoint_dir = f"ppo_dreamer/checkpoints/{name_wandb}"
        print(name_wandb)

        train_rewards = []
        test_rewards = []
        policy_losses = []
        value_losses = []

        actor, critic = create_agent(HIDDEN_DIMENSIONS, DROPOUT, INPUT_FEATURES, ACTOR_OUTPUT_FEATURES, NUM_LAYERS)
        optimizer_actor = optim.Adam(actor.parameters(), lr=LEARNING_RATE_ACTOR)
        optimizer_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE_CRITIC)
        optim.Adam(critic.parameters(), lr=LEARNING_RATE_CRITIC)

        start_episode = 0
        # Carregar checkpoint
        if use_checkpoint:
            checkpoint_info = load_checkpoint(
                checkpoint_path='ppo_dreamer/checkpoints/checkpoint_teste1_epoch117.pt',
                actor=actor,
                critic=critic,
                optimizer_actor=optimizer_actor,
                optimizer_critic=optimizer_critic
            )

            start_episode = checkpoint_info['episode'] + 1

        if use_wandb:
            wandb.init(project="ppo", name= name_wandb, config={
                "LATENT_DIM": LATENT_DIM,
                "IN_CHANNELS": IN_CHANNELS,
                "HIDDEN_UNITS": HIDDEN_UNITS,
                "INPUT_FEATURES": INPUT_FEATURES,
                "ACTOR_OUTPUT_FEATURES": ACTOR_OUTPUT_FEATURES,
                "MAX_EPISODES": MAX_EPISODES,
                "DISCOUNT_FACTOR": DISCOUNT_FACTOR,
                "REWARD_THRESHOLD": REWARD_THRESHOLD,
                "PRINT_INTERVAL": PRINT_INTERVAL,
                "PPO_STEPS": PPO_STEPS,
                "N_TRIALS": N_TRIALS,
                "EPSILON": EPSILON,
                "ENTROPY_COEFFICIENT": ENTROPY_COEFFICIENT,
                "HIDDEN_DIMENSIONS": HIDDEN_DIMENSIONS,
                "DROPOUT": DROPOUT,
                "LEARNING_RATE_ACTOR": LEARNING_RATE_ACTOR,
                "LEARNING_RATE_CRITIC": LEARNING_RATE_CRITIC
            },
            mode="online",           # Sincroniza apenas com a nuvem (sem arquivos locais)
            save_code=False,)         # Não salvar códigos localmente)

        for episode in range(start_episode, MAX_EPISODES+1):
            train_reward, states, actions, actions_log_probability, advantages, returns = forward_pass(
                    env_train,
                    actor,
                    critic,
                    encoder,
                    DISCOUNT_FACTOR)
            BATCH_SIZE = get_batch_size(returns.shape[0])
            policy_loss, value_loss, total_value_pred, total_log_prob_old, total_log_prob_new, total_surrogate_loss, total_surrogate_loss_1, total_surrogate_loss_2, total_policy_ratio, entropy  = update_policy(
                    actor,
                    critic,
                    states,
                    encoder,
                    actions,
                    actions_log_probability,
                    advantages,
                    returns,
                    optimizer_actor,
                    optimizer_critic,
                    optimizer_encoder,
                    PPO_STEPS,
                    EPSILON,
                    ENTROPY_COEFFICIENT,
                    BATCH_SIZE)
            test_reward = evaluate(env_test, actor, encoder, device)

            policy_losses.append(policy_loss)
            value_losses.append(value_loss)
            train_rewards.append(train_reward)
            test_rewards.append(test_reward)
            print(f"Episode: {episode} | Training Reward: {train_reward:.1f} | Testing Reward: {test_reward:.1f} | Loss Policy: {policy_loss:.2f} | Loss Value: {value_loss:.2f} |")

            if use_wandb is True:
                wandb.log({
                    "policy_loss": policy_loss,
                    "value_loss": value_loss,
                    "train_rewards": train_reward,
                    "test_rewards": test_reward,
                    "total_value_pred": total_value_pred,
                    "total_log_prob_old": total_log_prob_old,
                    "total_log_prob_new": total_log_prob_new,
                    "total_surrogate_loss": total_surrogate_loss,
                    "total_surrogate_loss_1": total_surrogate_loss_1,
                    "total_surrogate_loss_2": total_surrogate_loss_2,
                    "total_policy_ratio": total_policy_ratio, 
                    "entropy": entropy
                })

            if episode % PRINT_INTERVAL == 0:
                print(f'Episode: {episode:3d} | Training Reward: {train_reward:.1f} | '
                    f'Testing Reward: {test_reward:.1f} | '
                    f' Policy Loss: {policy_loss:.2f} | '
                    f'Value Loss: {value_loss:.2f} | ')
                
            if (train_reward >= REWARD_THRESHOLD) and (train_reward > best_reward) and (save_model is True):  
                best_reward = train_reward
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch{episode}.pt")

                torch.save({
                    'episode': episode,
                    'actor_state_dict': actor.state_dict(), 
                    'critic_state_dict': critic.state_dict(),  
                    'actor_optimizer_state_dict': optimizer_actor.state_dict(),  
                    'critic_optimizer_state_dict': optimizer_critic.state_dict(),
                    'policy_loss': policy_loss,
                    'value_loss': value_loss,
                    'reward': train_reward, 
                    'test_reward': test_reward,
                }, checkpoint_path)

run_ppo() 