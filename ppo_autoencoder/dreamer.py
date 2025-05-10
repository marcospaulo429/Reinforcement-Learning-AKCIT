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
import torch.nn.functional as F
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from test_rssm import RewardModel, TransitionModel

from stable_baselines3.common.atari_wrappers import (  # isort:skip
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)


@dataclass
class Args:
    vis_interval: int = 10
    """the interval of visualization"""
    latent_dim: int = 128
    """latent dimension of the VAE"""
    horizon: int = 3
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
    belief_size = 50        # Tamanho do vetor de crença
    hidden_size = 50       # Tamanho das camadas escondidas
    future_rnn = True       # Modelar o futuro com RNN
    mean_only = True        # Usar apenas a média da crença
    min_stddev = 1e-6       # Desvio padrão mínimo
    num_layers = 2          # Número de camadas nas redes neurais

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
    recon_coef: float = 10e-4
    """coefficient of the reconstruction loss"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    beta_kl_wm: float = 0.5
    target_kl: float = None
    """the target KL divergence threshold"""

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

# ----- Classe Encoder -----
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
        x = self.encoder_cnn(x / 255.0)
        mu = self.encoder_fc_mu(x)
        logvar = self.encoder_fc_logvar(x)
        return mu, logvar

# ----- Classe Decoder -----
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

    
def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

# Função de perda VAE
def vae_loss(recon_x, x, mu, logvar):
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl


class Agent(nn.Module):
    def __init__(self, latent_dim, envs):
        super().__init__()
        self.actor = layer_init(nn.Linear(latent_dim, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(latent_dim, 1), std=1)

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"dreamer_{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
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

    #device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    device = torch.device("mps" if torch.backends.mps.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(args.latent_dim, envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-4)

    encoder = Encoder(args.latent_dim).to(device)
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=args.learning_rate, eps=1e-4)


    action_dim = 1 if len(envs.single_action_space.shape) == 0 else envs.single_action_space.shape[0]
    transition_model = TransitionModel(args.latent_dim, args.belief_size, args.hidden_size, args.future_rnn, action_dim ,args.mean_only, args.min_stddev, args.num_layers).to(device)
    transition_model_optim = optim.Adam(transition_model.parameters(), lr=args.learning_rate, eps=1e-4)

    decoder = Decoder(args.latent_dim + args.belief_size).to(device)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=args.learning_rate, eps=1e-4)

    reward_model = RewardModel(args.hidden_size, args.latent_dim).to(device)
    reward_optim = optim.Adam(reward_model.parameters(), lr=args.learning_rate, eps=1e-4)

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
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done


            # ALGO LOGIC: action logic
            with torch.no_grad():
                mu, logvar = encoder.forward(next_obs/255.0)
                next_obs_latent = reparameterize(mu, logvar)
                action, logprob, _, value = agent.get_action_and_value(next_obs_latent)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward, dtype=torch.float32).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # bootstrap value if not done
        with torch.no_grad():
            mu, logvar = encoder.forward(next_obs / 255.0)
            next_obs_latent = reparameterize(mu, logvar)
            next_value = agent.get_value(next_obs_latent).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # Flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)

        b_logprobs = logprobs.reshape(-1)

        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)

        b_advantages = advantages.reshape(-1)

        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        prev_state = {
            'sample': torch.randn(args.minibatch_size, args.latent_dim, device=device),
            'rnn_state': torch.zeros(args.minibatch_size, args.hidden_size, device=device)
        }


        b_inds = np.arange(args.batch_size)
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                # DYNAMICS LEARNING: TODAS AS DIMENSOES VALIDADAS

                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                mu, logvar = encoder.forward(b_obs[mb_inds] / 255.0)

                next_obs_latent = reparameterize(mu, logvar)


                prev_state['sample'] = next_obs_latent
                posterior = transition_model._posterior(prev_state, b_actions[mb_inds].unsqueeze(-1), next_obs_latent)
                prior = transition_model._transition(prev_state, b_actions[mb_inds].unsqueeze(-1))
                

                kl_wm = transition_model.divergence_from_states(posterior, prior)
                features = transition_model.features_from_state(posterior)
                obs_recon = decoder(features)

                reconstruction_loss_fn = nn.MSELoss()
                reconstruction_loss = reconstruction_loss_fn(obs_recon, b_obs[mb_inds])

                loss_world_model = reconstruction_loss + (args.beta_kl_wm * kl_wm).sum()

                encoder_optimizer.zero_grad()
                decoder_optimizer.zero_grad()
                transition_model_optim.zero_grad()
                loss_world_model.backward()
                nn.utils.clip_grad_norm_(encoder.parameters(), args.max_grad_norm)
                nn.utils.clip_grad_norm_(decoder.parameters(), args.max_grad_norm)
                nn.utils.clip_grad_norm_(transition_model.parameters(), args.max_grad_norm)
                encoder_optimizer.step()
                decoder_optimizer.step()
                transition_model_optim.step()

                # BEHAVIOR LEARNING

                bh_traj = []
                bh_rewards = []
                bh_values = []

                prev_action = b_actions.long()[mb_inds]

                prev_state = {
                    'sample': torch.randn(args.minibatch_size, args.latent_dim, device=device),
                    'rnn_state': torch.zeros(args.minibatch_size, args.hidden_size, device=device)
                }

                with torch.no_grad():
                    state = transition_model._posterior(prev_state, prev_action.unsqueeze(-1), next_obs_latent.detach())

                for t in range(args.horizon):
                    # 1. Ação latente
                    action, newlogprob, entropy, value = agent.get_action_and_value(next_obs_latent.detach(), prev_action)

                    # 2. Transição (imaginação)
                    with torch.no_grad():
                        state = transition_model._transition(state, action.unsqueeze(-1))

                    # 3. Recompensa imaginada
                    with torch.no_grad():
                        reward = reward_model.forward(state['belief'], state['sample'])
                    next_obs_latent = state['sample']

                    # 4. Armazena
                    bh_traj.append(state)
                    bh_rewards.append(reward)
                    bh_values.append(value)

                # Empilha para [H, B]
                bh_rewards = torch.stack(bh_rewards, dim=0)
                bh_values = torch.stack(bh_values, dim=0)

                # Cálculo de V_lambda
                lambda_ = args.gae_lambda
                H = args.horizon
                gamma = args.gamma

                H, B, _ = bh_rewards.shape  # horizonte, batch, 1

                V_lambda = torch.zeros([B,1], device=device)

                #TODO: fazer verificacao do horizon
                # Loop de 1 até H-1
                for n in range(1, H):
                    discounts = gamma ** torch.arange(n, device=device).float()       
                    discounts = discounts.unsqueeze(-1)

                    Rn = (discounts * bh_rewards[:n].squeeze(-1)).sum(dim=0)                             

                    # 4) Calcula Vn = Rn + γⁿ·values[n] → ambas formas [B,1]
                    Vn = Rn + (gamma ** n) * bh_values[n].squeeze(-1)                                  # [B,1]

                    # 5) Peso do termo n
                    weight = (1 - lambda_) * (lambda_ ** (n - 1))                       # escalar

                    # 6) Acumula em V_lambda
                    V_lambda = V_lambda + weight * Vn  

                # Aplique o mesmo processo para o último passo (H)
                discounts = gamma ** torch.arange(H, device=device).float()  # [H]
                discounts = discounts.unsqueeze(-1)
                RH = (discounts * bh_rewards.squeeze(-1)).sum(0)  # [batch, 1]
                VH = RH + (gamma ** H) * bh_values[-1].squeeze(-1)  # [batch, 1]

                # Atualizando V_lambda com o último valor de Vn (VH)
                V_lambda += (lambda_ ** (H - 1)) * VH  # [batch, 1]

                # Último termo: λ^{H-1} * V^H
                discounts = gamma ** torch.arange(H, device=device).float()      # [H]
                RH = (discounts.unsqueeze(-1) * bh_rewards.squeeze(-1)).sum(0)                    # [B]
                VH = RH + (gamma ** H) * bh_values[-1]                              # [B]
                V_lambda += (lambda_ ** (H - 1)) * VH                            # [B]

                # Losses
                loss_actor = -V_lambda.mean()
                loss_critic = 0.5 * ((bh_values[0] - V_lambda) ** 2).mean()
                loss = loss_actor + loss_critic

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            # Log losses e métricas relevantes para Dreamer
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)

            writer.add_scalar("losses/world_model_loss", loss_world_model.item(), global_step)
            writer.add_scalar("losses/actor_loss", loss_actor.item(), global_step)
            writer.add_scalar("losses/critic_loss", loss_critic.item(), global_step)

            # (Opcional) Adicione separadamente a perda de reconstrução e KL se quiser analisar
            writer.add_scalar("losses/reconstruction_loss", reconstruction_loss.item(), global_step)
            writer.add_scalar("losses/kl_divergence", kl_wm.sum(dim=-1).mean().item(), global_step)


        # Velocidade de treinamento
        sps = int(global_step / (time.time() - start_time))
        print("SPS:", sps)
        writer.add_scalar("charts/SPS", sps, global_step)

        writer.add_scalar("diagnostics/value_mean", bh_values[0].mean().item(), global_step)
        writer.add_scalar("diagnostics/V_lambda_mean", V_lambda.mean().item(), global_step)
        writer.add_scalar("diagnostics/imagined_reward_mean", bh_rewards.mean().item(), global_step)
        writer.add_scalar("diagnostics/latent_std_mean", state['sample'].std().item(), global_step)



    envs.close()
    writer.close()