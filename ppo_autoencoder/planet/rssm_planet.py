# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_ataripy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from rssm import RewardModel, TransitionModel

from stable_baselines3.common.atari_wrappers import (  # isort:skip
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)


@dataclass
class Args:
    latent_dim: int = 512
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "BreakoutNoFrameskip-v4"
    """the id of the environment"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 8
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.1
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""
    recon_coef: float = 1e-6

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)
        return env

    return thunk

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Encoder(nn.Module):
    def __init__(self, latent_dim=32):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim

        # Camadas convolucionais
        self.encoder_cnn = nn.Sequential(
            layer_init(nn.Conv2d(4, 16, kernel_size=8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(16, 32, kernel_size=4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 32, kernel_size=3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Camadas fully connected para mu e logvar
        self.encoder_fc_mu = layer_init(nn.Linear(32 * 7 * 7, latent_dim))
        self.encoder_fc_logvar = layer_init(nn.Linear(32 * 7 * 7, latent_dim))

    def forward(self, x):
        x = self.encoder_cnn(x)
        mu = self.encoder_fc_mu(x)
        logvar = self.encoder_fc_logvar(x)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim=32):
        super(Decoder, self).__init__()

        # Camada fully connected
        self.decoder_fc = layer_init(nn.Linear(latent_dim, 32 * 7 * 7))
        
        # Camadas deconvolucionais (transpostas)
        self.decoder_deconv = nn.Sequential(
            nn.ReLU(),
            nn.Unflatten(1, (32, 7, 7)),
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 4, kernel_size=8, stride=4),
            nn.Sigmoid()
        )

    def forward(self, z):
        x = self.decoder_fc(z)
        return self.decoder_deconv(x)

# Função de perda VAE
def vae_loss(recon_x, x, mu, logvar):
    recon_loss = nn.functional.binary_cross_entropy(recon_x, x, reduction='mean')
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl

def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    if torch.cuda.is_available():
        device = torch.device("cuda")

    elif torch.backends.mps.is_available():
        device = torch.device("mps")

    else:
        device = torch.device("cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    print(envs.single_action_space.n)

    rssm = TransitionModel(args.latent_dim, args.belief_size, args.hidden_size, args.future_rnn, args.action_dim, args.mean_only, args.min_stddev, args.num_layers)
    rssm_optimizer = optim.Adam(rssm.parameters(), lr=args.learning_rate)

    reward_model = RewardModel(args.hidden_dim, args.state_dim)
    reward_optimizer = optim.Adam(reward_model.parameters(), lr=args.learning_rate)
    
    encoder = Encoder(args.latent_dim).to(device)
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=args.learning_rate)

    decoder = Decoder(args.latent_dim).to(device)   
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=1e-4)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        # Data collection and model fitting
        hidden_state = torch.zeros(args.num_envs, args.rssm_hidden_size, device=device)
        prev_action = torch.zeros(args.num_envs, args.action_dim, device=device)
        belief_state = None
        
        # Listas para acumular transições da iteração atual
        episode_obs = []
        episode_actions = []
        episode_next_obs = []
        episode_rewards = []
        episode_dones = []

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                mu, logvar = encoder(next_obs/255.0)
                next_obs_latent, hidden_state, belief_state = rssm.update_belief(
                    next_obs/255.0, 
                    prev_action, 
                    hidden_state,
                    belief_state
                )
                
                action = planner(
                    belief_state=belief_state,
                    hidden_state=hidden_state,
                    num_samples=args.cem_num_samples,
                    horizon=args.planning_horizon,
                    top_k=args.cem_top_k
                )
                
                if args.exploration_noise > 0:
                    noise = torch.randn_like(action) * args.exploration_noise
                    action = torch.clamp(action + noise, -1, 1)
            
            actions[step] = action
            prev_action = action.clone()
            
            # Action Repeat
            total_reward = 0
            for k in range(args.action_repeat):
                next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
                total_reward += reward
                if terminations.any() or truncations.any():
                    break
            
            # Armazenar transição atual
            episode_obs.append(next_obs_latent)
            episode_actions.append(action)
            episode_next_obs.append(next_obs.clone())
            episode_rewards.append(torch.tensor(total_reward, dtype=torch.float32, device=device))
            episode_dones.append(torch.tensor(
                np.logical_or(terminations, truncations), dtype=torch.float32, device=device))
            
            # Atualizar para próximo passo
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(
                np.logical_or(terminations, truncations)).to(device)

            # Logging
            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # Model Fitting com os dados coletados nesta iteração
        if len(episode_obs) > 0:  # Se coletamos algum dado
            # Converter listas para tensores
            b_obs = torch.stack(episode_obs)
            b_actions = torch.stack(episode_actions)
            b_next_obs = torch.stack(episode_next_obs)
            b_rewards = torch.stack(episode_rewards)
            b_dones = torch.stack(episode_dones)
            
            # Dimensões: [seq_len, num_envs, ...] -> [num_envs, seq_len, ...]
            b_obs = b_obs.transpose(0, 1)
            b_actions = b_actions.transpose(0, 1)
            b_next_obs = b_next_obs.transpose(0, 1)
            b_rewards = b_rewards.transpose(0, 1)
            b_dones = b_dones.transpose(0, 1)
            
            for epoch in range(args.epochs_model_fitting):
                # Processar cada ambiente separadamente
                for env_idx in range(args.num_envs):
                    # Resetar estados ocultos para este ambiente
                    env_hidden_state = torch.zeros(1, args.rssm_hidden_size, device=device)
                    env_prev_action = torch.zeros(1, args.action_dim, device=device)
                    env_belief_state = None
                    
                    # Listas para acumular losses
                    recon_losses = []
                    kl_losses = []
                    
                    # Processar cada passo temporal deste ambiente
                    for t in range(len(episode_obs) // args.num_envs):
                        # 1. Codificar observação atual
                        mu, logvar = encoder(b_obs[env_idx, t].unsqueeze(0)/255.0)
                        posterior_dist = torch.distributions.Normal(mu, logvar.exp().sqrt())
                        
                        # 2. Atualizar crença (RSSM)
                        latent_state, env_hidden_state, env_belief_state = rssm.update_belief(
                            b_obs[env_idx, t].unsqueeze(0)/255.0, 
                            env_prev_action, 
                            env_hidden_state,
                            env_belief_state
                        )
                        
                        # 3. Obter prior (transição)
                        prior_mu, prior_logvar = rssm.transition_model(env_hidden_state, env_prev_action)
                        prior_dist = torch.distributions.Normal(prior_mu, prior_logvar.exp().sqrt())
                        
                        # 4. Calcular termo de reconstrução
                        recon_mu, recon_logvar = decoder(latent_state)
                        recon_dist = torch.distributions.Normal(recon_mu, recon_logvar.exp().sqrt())
                        recon_loss = -recon_dist.log_prob(b_next_obs[env_idx, t].unsqueeze(0)/255.0).mean()
                        
                        # 5. Calcular termo KL
                        kl_loss = torch.distributions.kl_divergence(posterior_dist, prior_dist).mean()
                        
                        # Acumular losses
                        recon_losses.append(recon_loss)
                        kl_losses.append(kl_loss)
                        
                        # Atualizar ação anterior
                        env_prev_action = b_actions[env_idx, t].unsqueeze(0)
                    
                    # Calcular losses médias para este ambiente
                    if recon_losses:  # Se houve dados para processar
                        avg_recon_loss = torch.stack(recon_losses).mean()
                        avg_kl_loss = torch.stack(kl_losses).mean()
                        total_loss = avg_recon_loss + args.kl_weight * avg_kl_loss
                        
                        # Backpropagation
                        optimizer.zero_grad()
                        total_loss.backward()
                        torch.nn.utils.clip_grad_norm_(parameters, args.max_grad_norm)
                        optimizer.step()
                        
                        # Logging (apenas no último epoch)
                        if epoch == args.epochs_model_fitting - 1 and iteration % args.log_interval == 0:
                            writer.add_scalar("losses/total_loss", total_loss.item(), global_step)
                            writer.add_scalar("losses/recon_loss", avg_recon_loss.item(), global_step)
                            writer.add_scalar("losses/kl_loss", avg_kl_loss.item(), global_step)

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        with torch.no_grad():
            sample_obs = b_obs[:10].to(device) / 255.0 
            mu, logvar = encoder(sample_obs)
            z = reparameterize(mu, logvar)
            recon = decoder(z)

            # Pegando só o último canal e repetindo para RGB
            sample_obs_last = sample_obs[:, -1].unsqueeze(1)       
            recon_last = recon[:, -1].unsqueeze(1)                 

            sample_obs_rgb = sample_obs_last.repeat(1, 3, 1, 1)    
            recon_rgb = recon_last.repeat(1, 3, 1, 1)             


            writer.add_images("reconstructions/original", sample_obs_rgb, global_step)
            writer.add_images("reconstructions/reconstructed", recon_rgb, global_step)

    envs.close()
    writer.close()