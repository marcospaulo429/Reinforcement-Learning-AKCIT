import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical 
import numpy as np

@torch.no_grad() 
def imagine_trajectories(initial_latent_states, transition_model, reward_model, agent, args):
    """
    Imagina trajetórias futuras usando o modelo de mundo (World Model) e a política do agente.
    As trajetórias imaginadas são usadas para treinar o ator e o crítico.

    Args:
        initial_latent_states (torch.Tensor): O estado latente inicial (sample) a partir do qual a imaginação começa.
                                               No Dreamer, estas são tipicamente amostras de estados posteriores
                                               inferidos a partir de dados reais. Shape: [batch_size, latent_dim]
        transition_model: O componente de transição do modelo de mundo (tipicamente um RSSM).
        reward_model: O componente de previsão de recompensa do modelo de mundo.
        agent: A instância do agente que contém as redes de ator (política) e crítico (valor).
        args: Objeto contendo vários hiperparâmetros (por exemplo, horizon_to_imagine, action_dim,
              belief_size, latent_dim).

    Returns:
        imagined_states (torch.Tensor): Estados latentes imaginados (samples). Shape: [total_imagined_steps, latent_dim]
        imagined_actions (torch.Tensor): Ações imaginadas pela política.
                                        Shape: [total_imagined_steps, action_dim] (para ações contínuas/one-hot)
                                        ou [total_imagined_steps] (para ações discretas sem one-hot).
        imagined_rewards (torch.Tensor): Recompensas previstas pelo modelo de mundo para os estados imaginados.
                                         Shape: [total_imagined_steps, 1]
        imagined_values (torch.Tensor): Valores previstos pela rede de crítico para os estados imaginados.
                                        Shape: [total_imagined_steps, 1]
        imagined_log_probs (torch.Tensor): Log-probabilidades das ações imaginadas pela política.
                                           Shape: [total_imagined_steps, 1]
    """
    batch_size = initial_latent_states.shape[0]
    device = initial_latent_states.device

    imagined_states_list = []
    imagined_actions_list = []
    imagined_rewards_list = []
    imagined_values_list = []
    imagined_log_probs_list = []

    prev_state_wm = {
        'sample': initial_latent_states,
        'rnn_state': torch.zeros(batch_size, args.belief_size, device=device), 
        'mean': initial_latent_states, 
        'stddev': torch.ones_like(initial_latent_states), 
        'belief': torch.zeros(batch_size, args.belief_size, device=device) 
    }

    for t in range(args.horizon_to_imagine):
        action, log_prob, _, value = agent.get_action_and_value(prev_state_wm['sample'])

        if action.dtype == torch.long:
            action_one_hot = F.one_hot(action, num_classes=args.action_dim).float()
        else:
            action_one_hot = action

        prior_state_wm = transition_model._transition(prev_state_wm, action_one_hot)

        predicted_reward_wm = reward_model(prior_state_wm['belief'], prior_state_wm['sample'])

        imagined_states_list.append(prior_state_wm['sample'])
        imagined_actions_list.append(action) 
        imagined_rewards_list.append(predicted_reward_wm)
        imagined_values_list.append(value)
        imagined_log_probs_list.append(log_prob)

        prev_state_wm = prior_state_wm

    imagined_states = torch.cat(imagined_states_list, dim=0)
    imagined_actions = torch.cat(imagined_actions_list, dim=0)
    imagined_rewards = torch.cat(imagined_rewards_list, dim=0)
    imagined_values = torch.cat(imagined_values_list, dim=0)
    imagined_log_probs = torch.cat(imagined_log_probs_list, dim=0)

    return imagined_states, imagined_actions, imagined_rewards, imagined_values, imagined_log_probs


def behavior_learning(args, obs_latents_wm_tomodel, agent, optimizer, imagine_trajectories, device, transition_model, reward_model):

    obs_latents_wm_tomodel = obs_latents_wm_tomodel.detach()

    actor_losses = []
    critic_losses = []
    total_behavior_losses = []
    entropies = []
    clipfracs = []
    approx_kls = []

    for epoch in range(args.update_epochs):
        imagined_states, imagined_actions, imagined_rewards, imagined_values, imagined_log_probs = \
            imagine_trajectories(
                initial_latent_states=obs_latents_wm_tomodel, 
                transition_model=transition_model,
                reward_model=reward_model,
                agent=agent,
                args=args
            )

        
        imagined_rewards_flat = imagined_rewards.squeeze(-1)
        imagined_values_flat = imagined_values.squeeze(-1)

        num_initial_sequences = obs_latents_wm_tomodel.shape[0] 
        
        imagined_values_reshaped_for_lambda = imagined_values_flat.view(num_initial_sequences, args.horizon_to_imagine)
        imagined_rewards_reshaped_for_lambda = imagined_rewards_flat.view(num_initial_sequences, args.horizon_to_imagine)

        lambda_returns_reshaped = torch.zeros_like(imagined_rewards_reshaped_for_lambda, device=device)
        
        next_value_bootstrap = imagined_values_reshaped_for_lambda[:, -1]
        
        for t in reversed(range(args.horizon_to_imagine)):
            if t == args.horizon_to_imagine - 1:
                current_lambda_return_step = imagined_rewards_reshaped_for_lambda[:, t] + \
                                             args.lambda_return_gamma * next_value_bootstrap
            else:
                next_imagined_value = imagined_values_reshaped_for_lambda[:, t+1]
                next_lambda_return_step = lambda_returns_reshaped[:, t+1]

                current_lambda_return_step = imagined_rewards_reshaped_for_lambda[:, t] + \
                                              args.lambda_return_gamma * ( (1 - args.lambda_return_lambda) * next_imagined_value + \
                                                                           args.lambda_return_lambda * next_lambda_return_step )
            lambda_returns_reshaped[:, t] = current_lambda_return_step
        
        lambda_returns_flat = lambda_returns_reshaped.flatten()
        
        lambda_returns_flat = (lambda_returns_flat - lambda_returns_flat.mean()) / (lambda_returns_flat.std() + 1e-8)

        total_imagined_steps = imagined_states.shape[0] 
        mb_inds_behavior = np.arange(total_imagined_steps)
        np.random.shuffle(mb_inds_behavior) 

        for start in range(0, total_imagined_steps, args.minibatch_size):
            end = start + args.minibatch_size
            current_mb_inds = mb_inds_behavior[start:end]

            mb_imagined_states = imagined_states.index_select(0, torch.tensor(current_mb_inds, device=device))
            mb_imagined_actions = imagined_actions.index_select(0, torch.tensor(current_mb_inds, device=device))
            mb_imagined_log_probs = imagined_log_probs.index_select(0, torch.tensor(current_mb_inds, device=device))
            mb_imagined_values = imagined_values_flat.index_select(0, torch.tensor(current_mb_inds, device=device))
            mb_lambda_returns = lambda_returns_flat.index_select(0, torch.tensor(current_mb_inds, device=device))


            logits = agent.actor(mb_imagined_states)
            new_action_dist = Categorical(logits=logits) 
            
            newlogprob = new_action_dist.log_prob(mb_imagined_actions).unsqueeze(-1)
            
            entropy = new_action_dist.entropy().mean()
            
            newvalue = agent.critic(mb_imagined_states).view(-1)

            logratio = newlogprob - mb_imagined_log_probs
            ratio = logratio.exp()
            with torch.no_grad(): 
                old_approx_kl = (-logratio).mean() 
                approx_kl = ((ratio - 1) - logratio).mean() 
                clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())
            
            approx_kls.append(approx_kl.item()) 

            actor_loss = - (newlogprob * mb_lambda_returns).mean()
            actor_loss = actor_loss - args.ent_coef * entropy

            if args.clip_vloss:
                v_loss_unclipped = (newvalue - mb_lambda_returns) ** 2
                v_clipped = mb_imagined_values + torch.clamp(
                    newvalue - mb_imagined_values,
                    -args.clip_coef, 
                    args.clip_coef,
                )
                v_loss_clipped = (v_clipped - mb_lambda_returns) ** 2
                v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
            else:
                v_loss = 0.5 * ((newvalue - mb_lambda_returns) ** 2).mean()

            total_behavior_loss = actor_loss + v_loss * args.vf_coef

            optimizer.zero_grad() 
            total_behavior_loss.backward() 
            nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm) 
            optimizer.step() 

            actor_losses.append(actor_loss.item())
            critic_losses.append(v_loss.item())
            total_behavior_losses.append(total_behavior_loss.item())
            entropies.append(entropy.item())

    return agent, actor_losses, critic_losses, total_behavior_losses, entropies, clipfracs, approx_kls
