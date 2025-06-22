import torch
import torch.nn as nn
import torch.nn.functional as F
from models import reparameterize

def dynamics_learning(args, transition_model, reward_model, encoder, decoder, 
                      transition_optimizer, reward_optimizer, encoder_optimizer, 
                      decoder_optimizer, obs_seq, actions_seq, rewards_seq, device, vae_loss):

    # 1. Define initial states for the World Model for EACH ENVIRONMENT (num_envs)
    initial_belief = torch.zeros(args.num_envs, transition_model.belief_size).to(device)
    initial_state_sample = torch.zeros(args.num_envs, transition_model.latent_dim).to(device)

    prev_state_wm = {
        'sample': initial_state_sample,
        'rnn_state': initial_belief,
        'mean': initial_state_sample,
        'stddev': torch.ones_like(initial_state_sample), # Small value to avoid stddev zero
        'belief': initial_belief
    }

    # Lists to collect states and reward predictions at each time step
    prior_states_wm_list = []
    posterior_states_wm_list = []
    predicted_rewards_wm_list = []

    # No "previous action" at the very first step of the sequence
    zero_action_ids = torch.zeros(1, args.num_envs, dtype=torch.long, device=device) 
    
    actions_for_posterior_ids = torch.cat([zero_action_ids, actions_seq], dim=0)
    actions_for_prior_ids = actions_seq

    actions_for_posterior_one_hot = F.one_hot(actions_for_posterior_ids, num_classes=args.action_dim).float() 
    actions_for_prior_one_hot = F.one_hot(actions_for_prior_ids, num_classes=args.action_dim).float()      

    # Reshape observations for the encoder: [num_steps * num_envs, C, H, W]
    obs_seq = obs_seq.view(-1, *obs_seq.shape[2:]) 
    
    # Encode all flattened observations
    mu, logvar = encoder(obs_seq / 255)
    obs_latents_wm_tomodel = reparameterize(mu, logvar)
    
    # Reshape latents back to sequence format: [num_steps, num_envs, latent_dim]
    obs_latents_wm = obs_latents_wm_tomodel.view(args.num_steps, args.num_envs, args.latent_dim)

    for t in range(args.num_steps):
        current_obs_latent_wm = obs_latents_wm[t] # Shape: [num_envs, latent_dim]
        action_for_posterior = actions_for_posterior_one_hot[t] # Shape: [num_envs, action_dim]
        action_for_prior = actions_for_prior_one_hot[t]         # Shape: [num_envs, action_dim]

        posterior_state_wm = transition_model._posterior(prev_state_wm, action_for_posterior, current_obs_latent_wm)
        prior_state_wm = transition_model._transition(posterior_state_wm, action_for_prior)
        predicted_reward_wm = reward_model(posterior_state_wm['belief'], posterior_state_wm['sample'])

        prior_states_wm_list.append(prior_state_wm)
        posterior_states_wm_list.append(posterior_state_wm)
        predicted_rewards_wm_list.append(predicted_reward_wm)

        prev_state_wm = posterior_state_wm

    # Stack the collected states and rewards
    prior_states_wm_stacked = {k: torch.stack([s[k] for s in prior_states_wm_list], dim=0) for k in prior_states_wm_list[0]}
    posterior_states_wm_stacked = {k: torch.stack([s[k] for s in posterior_states_wm_list], dim=0) for k in posterior_states_wm_list[0]}
    predicted_rewards_wm_stacked = torch.cat(predicted_rewards_wm_list, dim=0).squeeze(-1)

    rewards_seq_flat = rewards_seq.flatten() # Shape: [num_steps * num_envs]

    # Calculate World Model losses
    # Assuming transition_model has a divergence_from_states method
    kl_loss_wm = transition_model.divergence_from_states(prior_states_wm_stacked, posterior_states_wm_stacked).mean()
    loss_reward_wm = F.mse_loss(predicted_rewards_wm_stacked, rewards_seq_flat)

    # Reconstruction loss
    recon_images = decoder.forward(obs_latents_wm_tomodel)
    recon_loss = vae_loss(recon_images, obs_seq / 255, mu, logvar)
    recon_loss = recon_loss.mean()

    # Total World Model loss
    total_world_model_loss = (kl_loss_wm * args.kl_beta + 
                              loss_reward_wm * args.reward_beta + 
                              recon_loss * args.recon_coef)

    # Backpropagation and optimization
    transition_optimizer.zero_grad()
    reward_optimizer.zero_grad()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    total_world_model_loss.backward()

    nn.utils.clip_grad_norm_(transition_model.parameters(), args.max_grad_norm)
    nn.utils.clip_grad_norm_(reward_model.parameters(), args.max_grad_norm)
    nn.utils.clip_grad_norm_(encoder.parameters(), args.max_grad_norm)
    nn.utils.clip_grad_norm_(decoder.parameters(), args.max_grad_norm)

    decoder_optimizer.step()
    transition_optimizer.step()
    reward_optimizer.step()
    encoder_optimizer.step()

    return  encoder, decoder, transition_model, reward_model, total_world_model_loss, kl_loss_wm, loss_reward_wm, recon_loss, obs_latents_wm_tomodel