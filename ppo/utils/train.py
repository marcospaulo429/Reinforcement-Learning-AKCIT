import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent)) 

import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from utils.auxiliares import converter_cinza, training_device
device = training_device()
from models.autoencoder import reparameterize
from utils.ppo_functions import (
    calculate_returns,
    calculate_advantages,
    calculate_surrogate_loss,
    calculate_losses,
    init_training
)

def forward_pass(env, actor, critic, encoder, discount_factor):
    encoder.train()
    raw_states, actions, actions_log_probability, values, rewards, done, episode_reward = init_training()
    time_step = env.reset()

    raw_state = converter_cinza(time_step.observation['pixels'])
    raw_state = raw_state.astype(np.float32) / 127.5 - 1.0 
    raw_state = torch.tensor(raw_state, dtype=torch.float32).to(device)
    raw_state = raw_state.unsqueeze(0).unsqueeze(0)
    
    z_mean, z_log_var = encoder(raw_state)

    state = reparameterize(z_mean, z_log_var).to(device)

    actor.train()
    critic.train()
    while not done:
        raw_states.append(raw_state)
        
        # Garante float32 nas saídas para trabalhar com mac
        action, log_prob_action, _ = actor(state)
        value_pred = critic(state).float() 
        
        time_step = env.step(action.item())
        done = time_step.last()
        reward = time_step.reward 
        
        # Processamento do novo estado
        raw_state = converter_cinza(time_step.observation['pixels'])
        raw_state = raw_state.astype(np.float32) / 127.5 - 1.0 
        raw_state = torch.tensor(raw_state, dtype=torch.float32).to(device)
        raw_state = raw_state.unsqueeze(0).unsqueeze(0)
        
        z_mean, z_log_var = encoder(raw_state)
        state = reparameterize(z_mean, z_log_var).to(device)

        actions.append(action)
        actions_log_probability.append(log_prob_action)
        values.append(value_pred)
        rewards.append(float(reward))  
        episode_reward += reward

    # Concatenação com float32
    states = torch.cat(raw_states).to(device)
    actions = torch.cat(actions).to(device)
    actions_log_probability = torch.cat(actions_log_probability).to(device)
    values = torch.cat(values).squeeze(-1).to(device)
    
    # Modificação crítica: Garante float32 nos retornos
    returns = calculate_returns(rewards, discount_factor).float().to(device)  # <-- Correção aqui
    advantages = calculate_advantages(returns, values).float().to(device)

    return episode_reward, states, actions, actions_log_probability, advantages, returns

def update_policy(
        actor,
        critic,
        raw_states,
        encoder,
        actions,
        actions_log_probability_old,
        advantages,
        returns,
        optimizer_actor,
        optimizer_critic,
        optimizer_encoder,
        ppo_steps,
        epsilon,
        entropy_coefficient,
        BATCH_SIZE):

    total_policy_loss = 0
    total_value_loss = 0
    total_value_pred = 0
    total_log_prob_old = 0
    total_log_prob_new = 0
    total_surrogate_loss = 0
    total_surrogate_loss_1 = 0
    total_surrogate_loss_2 = 0
    total_policy_ratio = 0
    total_entropy = 0

    actions_log_probability_old = actions_log_probability_old.detach()
    actions = actions.detach()

    training_results_dataset = TensorDataset(
            raw_states,
            actions,
            actions_log_probability_old,
            advantages,
            returns)

    batch_dataset = DataLoader(
            training_results_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False)

    for _ in range(ppo_steps):
        for batch_idx, (raw_states, actions, actions_log_probability_old, advantages, returns) in enumerate(batch_dataset):
            # get new log prob of actions for all input states
            z_mean, z_log_var = encoder(raw_states)
            states = reparameterize(z_mean, z_log_var)
            action, actions_log_probability_new, probability_distribution_new = actor(states)
            value_pred = critic(states)
            value_pred = value_pred.squeeze(-1)
            entropy = -actions_log_probability_new.mean() 

            # estimate new log probabilities using old actions
            actions_log_probability_new = probability_distribution_new.log_prob(actions)
            surrogate_loss, surrogate_loss_1, surrogate_loss_2, policy_ratio = calculate_surrogate_loss(
                    actions_log_probability_old,
                    actions_log_probability_new,
                    epsilon,
                    advantages)
            policy_loss, value_loss = calculate_losses(
                    surrogate_loss,
                    entropy,
                    entropy_coefficient,
                    returns,
                    value_pred)
            
              # Zerar gradientes
            #optimizer_encoder.zero_grad(set_to_none=True)
            optimizer_actor.zero_grad(set_to_none=True)
            optimizer_critic.zero_grad(set_to_none=True)

            # Backward único
            total_loss = policy_loss + value_loss
            total_loss.backward()

            # Atualizar todos os otimizadores APÓS o backward
            #optimizer_encoder.step()
            optimizer_actor.step()
            optimizer_critic.step()


            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_value_pred += value_pred.mean().item()
            total_log_prob_old += actions_log_probability_old.mean().item()
            total_log_prob_new += actions_log_probability_new.mean().item()
            total_surrogate_loss += surrogate_loss.mean().item()
            total_surrogate_loss_1 += surrogate_loss_1.mean().item()
            total_surrogate_loss_2 += surrogate_loss_2.mean().item()
            total_policy_ratio += policy_ratio.mean().item()
            total_entropy += entropy

    return total_policy_loss / ppo_steps, total_value_loss / ppo_steps, total_value_pred / ppo_steps, total_log_prob_old / ppo_steps, total_log_prob_new / ppo_steps, total_surrogate_loss / ppo_steps, total_surrogate_loss_1 / ppo_steps, total_surrogate_loss_2 / ppo_steps, total_policy_ratio / ppo_steps, total_entropy / ppo_steps

def evaluate(env, actor, encoder, device):
    actor.eval()
    
    time_step = env.reset()
    episode_reward = 0
    done = False
    
    # Processamento inicial do estado
    state = converter_cinza(time_step.observation['pixels'])
    state = state.astype(np.float32) / 127.5 - 1.0
    state = torch.tensor(state).float().to(device)  # Convertido para float e movido para o device
    state = state.unsqueeze(0).unsqueeze(0)
    
    with torch.no_grad():
        z_mean, z_log_var = encoder(state)
    state = reparameterize(z_mean, z_log_var).detach()

    while not done:
        with torch.no_grad():
            action, log_prob_action, dist = actor(state)

        time_step = env.step(action.item())
        done = time_step.last()
        reward = time_step.reward #if time_step.reward is not None else 0.0
        
        # Processamento do novo estado
        state = converter_cinza(time_step.observation['pixels'])
        state = state.astype(np.float32) / 127.5 - 1.0
        state = torch.tensor(state).float().to(device)  # Convertido para float e movido para o device
        state = state.unsqueeze(0).unsqueeze(0)
        
        with torch.no_grad():
            z_mean, z_log_var = encoder(state)
        state = reparameterize(z_mean, z_log_var).detach()
        
        episode_reward += reward

    return episode_reward
