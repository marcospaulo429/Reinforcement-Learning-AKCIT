import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical # Importe Categorical
import numpy as np

# A função imagine_trajectories.
# Assume que transition_model, reward_model, actor_critic, e args estão acessíveis
# ou são passados como argumentos.
@torch.no_grad() # Trajetórias imaginadas não geram gradientes para o World Model
def imagine_trajectories(initial_latent_states, encoder, transition_model, reward_model, agent, args):
    """
    Imagina trajetórias futuras usando o modelo de mundo e a política.

    Args:
        initial_latent_states (torch.Tensor): O estado latente inicial (sample) para a imaginação.
                                               Shape: [batch_size, latent_dim]
        encoder: O modelo de codificação de observações. (Não usado diretamente nesta função, mas passado para consistência)
        transition_model: O modelo de transição (RSSM).
        reward_model: O modelo de recompensa.
        actor_critic: A rede de ator (política) e crítico (valor).
        args: Objeto contendo os hiperparâmetros (como horizon_length, action_dim, belief_size, latent_dim).

    Returns:
        imagined_states (torch.Tensor): Estados latentes imaginados (samples). Shape: [batch_size * horizon_length, latent_dim]
        imagined_actions (torch.Tensor): Ações imaginadas pela política. Shape: [batch_size * horizon_length, action_dim] (se one-hot) ou [batch_size * horizon_length] (se discreta) ou [batch_size * horizon_length, action_dim] (se contínua)
        imagined_rewards (torch.Tensor): Recompensas previstas pelo modelo de mundo. Shape: [batch_size * horizon_length, 1]
        imagined_values (torch.Tensor): Valores previstos pela rede de crítico. Shape: [batch_size * horizon_length, 1]
        imagined_log_probs (torch.Tensor): Log-probabilidades das ações imaginadas. Shape: [batch_size * horizon_length, 1]
    """
    batch_size = initial_latent_states.shape[0]
    device = initial_latent_states.device

    imagined_states_list = []
    imagined_actions_list = []
    imagined_rewards_list = []
    imagined_values_list = []
    imagined_log_probs_list = []

    # Inicialização do prev_state_wm, consistente com seu treinamento do world model.
    prev_state_wm = {
        'sample': initial_latent_states,
        'rnn_state': torch.zeros(batch_size, args.belief_size, device=device),
        'mean': initial_latent_states,
        'stddev': torch.ones_like(initial_latent_states),
        'belief': torch.zeros(batch_size, args.belief_size, device=device)
    }

    # Loop de imaginação
    for t in range(args.horizon_to_imagine):

        action, log_prob, _, value = agent.get_action_and_value(prev_state_wm['sample'])

        # Converter a ação para one-hot se for discreta e seu transition_model espera
        if action.dtype == torch.long:
            action_one_hot = F.one_hot(action, num_classes=args.action_dim).float()
        else:
            action_one_hot = action 

        # 2. Transição no modelo de mundo (prior_state_wm)
        prior_state_wm = transition_model._transition(prev_state_wm, action_one_hot)

        # 3. Prever recompensa para o novo estado (prior_state_wm)
        predicted_reward_wm = reward_model(prior_state_wm['belief'], prior_state_wm['sample'])

        # Armazenar os resultados
        imagined_states_list.append(prior_state_wm['sample'])
        imagined_actions_list.append(action)
        imagined_rewards_list.append(predicted_reward_wm)
        imagined_values_list.append(current_value)
        imagined_log_probs_list.append(log_prob)

        # O próximo estado latente imaginado se torna o estado anterior para a próxima iteração
        prev_state_wm = prior_state_wm

    # Concatenar todos os tensores na dimensão do batch e da sequência
    imagined_states = torch.cat(imagined_states_list, dim=0)
    imagined_actions = torch.cat(imagined_actions_list, dim=0)
    imagined_rewards = torch.cat(imagined_rewards_list, dim=0)
    imagined_values = torch.cat(imagined_values_list, dim=0)
    imagined_log_probs = torch.cat(imagined_log_probs_list, dim=0)

    return imagined_states, imagined_actions, imagined_rewards, imagined_values, imagined_log_probs


def behavior_learning(args, obs_latents_wm_tomodel, actions, encoder, transition_model, 
                      reward_model, agent, optimizer, imagine_trajectories, device):
    """
    Performs one step of behavior learning (Actor and Critic optimization) in Dreamer.

    Args:
        args: An object containing various arguments/hyperparameters 
              (e.g., update_epochs, horizon_to_imagine, lambda_return_gamma, 
              lambda_return_lambda, minibatch_size, ent_coef, vf_coef, 
              clip_vloss, clip_coef, max_grad_norm).
        obs_latents_wm_tomodel (torch.Tensor): Encoded latent states from real data, 
                                             shape: [num_steps * num_envs, latent_dim].
        encoder: The observation encoder (needed by imagine_trajectories).
        transition_model: The world model's transition component (needed by imagine_trajectories).
        reward_model: The world model's reward prediction component (needed by imagine_trajectories).
        agent: Your instance of the Agent (Actor and Critic networks).
        optimizer: The optimizer for the Agent's parameters.
        imagine_trajectories (function): A function that simulates trajectories in the 
                                         world model's imagination.
        device (torch.device): The device (CPU or GPU) to run computations on.

    Returns:
        tuple: A tuple containing lists of losses for logging:
               - actor_losses (list): List of actor loss values per minibatch.
               - critic_losses (list): List of critic loss values per minibatch.
               - total_behavior_losses (list): List of total behavior loss values per minibatch.
               - entropies (list): List of entropy values per minibatch.
               - clipfracs (list): List of clip fractions (for PPO-like logging).
               - approx_kls (list): List of approximate KL divergences (for PPO-like logging).
    """

    # Detach latent states to prevent gradients from flowing back into the World Model
    # during behavior learning.
    obs_latents_wm_tomodel = obs_latents_wm_tomodel.detach()

    actor_losses = []
    critic_losses = []
    total_behavior_losses = []
    entropies = []
    clipfracs = []
    approx_kls = []

    # Loop for optimization epochs (multiple passes over imagined data)
    for epoch in range(args.update_epochs):
        # 1. Imagine Trajectories
        # Uses the current policy (`agent`) to generate actions in imagination.
        imagined_states, imagined_actions, imagined_rewards, imagined_values, imagined_log_probs = \
            imagine_trajectories(
                obs_latents_wm_tomodel,
                encoder,
                transition_model,
                reward_model,
                agent,
                args
            )

        # 2. Compute Value Estimates V_lambda(s_tau) (lambda-returns)
        # Flatten tensors for lambda-return calculation.
        imagined_rewards_flat = imagined_rewards.squeeze(-1)
        imagined_values_flat = imagined_values.squeeze(-1)
        
        # Reshape to (num_initial_sequences, horizon_length) for reverse calculation.
        num_initial_sequences = obs_latents_wm_tomodel.shape[0]
        
        imagined_values_reshaped_for_lambda = imagined_values_flat.view(num_initial_sequences, args.horizon_to_imagine)
        imagined_rewards_reshaped_for_lambda = imagined_rewards_flat.view(num_initial_sequences, args.horizon_to_imagine)

        lambda_returns_reshaped = torch.zeros_like(imagined_rewards_reshaped_for_lambda, device=device)
        
        # Bootstrap value from the last imagined state in EACH sequence.
        next_value_bootstrap = imagined_values_reshaped_for_lambda[:, -1]
        
        # Reverse calculation of lambda-return for each imagined sequence.
        for t in reversed(range(args.horizon_to_imagine)):
            if t == args.horizon_to_imagine - 1:
                # For the last step, use the bootstrapped value.
                current_lambda_return_step = imagined_rewards_reshaped_for_lambda[:, t] + \
                                             args.lambda_return_gamma * next_value_bootstrap
            else:
                # For other steps, use the value of the next state in imagination.
                next_imagined_value = imagined_values_reshaped_for_lambda[:, t+1]
                next_lambda_return_step = lambda_returns_reshaped[:, t+1]

                # Lambda-return formula (Equation 6 of Dreamer)
                current_lambda_return_step = imagined_rewards_reshaped_for_lambda[:, t] + \
                                              args.lambda_return_gamma * ( (1 - args.lambda_return_lambda) * next_imagined_value + \
                                                                           args.lambda_return_lambda * next_lambda_return_step )
            lambda_returns_reshaped[:, t] = current_lambda_return_step
        
        # Re-flatten to (total_imagined_steps).
        lambda_returns_flat = lambda_returns_reshaped.flatten()
        
        # Normalize returns (common for stability).
        lambda_returns_flat = (lambda_returns_flat - lambda_returns_flat.mean()) / (lambda_returns_flat.std() + 1e-8)

        # Prepare data for mini-batches for actor and critic optimizations.
        total_imagined_steps = imagined_states.shape[0] # num_initial_sequences * horizon_length
        mb_inds_behavior = np.arange(total_imagined_steps)
        np.random.shuffle(mb_inds_behavior) # Shuffle indices for mini-batches

        for start in range(0, total_imagined_steps, args.minibatch_size):
            end = start + args.minibatch_size
            current_mb_inds = mb_inds_behavior[start:end]

            # Get data for the minibatch from the imagined tensors.
            # Ensure indices are torch.tensor on the correct device
            mb_imagined_states = imagined_states.index_select(0, torch.tensor(current_mb_inds, device=device))
            mb_imagined_actions = imagined_actions.index_select(0, torch.tensor(current_mb_inds, device=device))
            mb_imagined_log_probs = imagined_log_probs.index_select(0, torch.tensor(current_mb_inds, device=device))
            mb_imagined_values = imagined_values_flat.index_select(0, torch.tensor(current_mb_inds, device=device))
            mb_lambda_returns = lambda_returns_flat.index_select(0, torch.tensor(current_mb_inds, device=device))

            # Get new actor/critic predictions for the minibatch states.
            new_action_dist = agent.actor(mb_imagined_states)
            newlogprob = new_action_dist.log_prob(mb_imagined_actions).unsqueeze(-1)
            entropy = new_action_dist.entropy().mean() # Entropy of the new policy
            newvalue = agent.critic(mb_imagined_states).view(-1) # Critic value under the new policy

            # Calculate PPO-like metrics for logging (optional).
            logratio = newlogprob - mb_imagined_log_probs
            ratio = logratio.exp()
            with torch.no_grad():
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())
            
            approx_kls.append(approx_kl.item()) # Store for logging

            # 3. Policy Loss (Actor Optimization) - Maximize V_lambda(s_tau)
            actor_loss = - (newlogprob * mb_lambda_returns).mean()
            # Add entropy term to encourage exploration.
            actor_loss = actor_loss - args.ent_coef * entropy

            # 4. Value Loss (Critic Optimization)
            if args.clip_vloss:
                v_loss_unclipped = (newvalue - mb_lambda_returns) ** 2
                v_clipped = mb_imagined_values + torch.clamp(
                    newvalue - mb_imagined_values,
                    -args.clip_coef, # Reusing clip_coef for value, if desired
                    args.clip_coef,
                )
                v_loss_clipped = (v_clipped - mb_lambda_returns) ** 2
                v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
            else:
                v_loss = 0.5 * ((newvalue - mb_lambda_returns) ** 2).mean()

            # --- Unified Optimization for Actor and Critic using a single `optimizer` ---
            # The total loss is the sum of actor (policy) and critic (value) losses
            # with their respective coefficients.
            total_behavior_loss = actor_loss + v_loss * args.vf_coef

            optimizer.zero_grad()
            total_behavior_loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
            optimizer.step()

            actor_losses.append(actor_loss.item())
            critic_losses.append(v_loss.item())
            total_behavior_losses.append(total_behavior_loss.item())
            entropies.append(entropy.item())

    return actor_losses, critic_losses, total_behavior_losses, entropies, clipfracs, approx_kls