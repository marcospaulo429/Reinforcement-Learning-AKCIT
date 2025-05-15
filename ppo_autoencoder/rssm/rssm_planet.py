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

def planner_discrete(env, H, J, batch, initial_obs, rssm): #TODO: colocar batch
    # Inicializar distribuição das ações (contínuas entre -1 e 1)
    mean = torch.zeros(H)
    std = torch.ones(H)
    best_total_reward = -float('inf')
    best_actions = None

    for i in range(I):
        # Gerar sequências de ações candidatas
        dist = Normal(mean, std)
        action_sequences = dist.sample((J,))  # Forma (J, H)
        action_sequences = torch.clamp(action_sequences, -1.0, 1.0)  # TODO: ver qual as acoes
        
        total_rewards = torch.zeros(J)

        initial_state = #TODO
        """

        # Codificar estado inicial
        initial_obs, _ = env.reset()
        initial_state = rssm.encode(initial_obs[None], None)  # [1, state_dim]

            for j in range(J):
        state = initial_state.clone()
        total_reward = 0.0
        
        for t in range(H):
            action = action_sequences[j, t].unsqueeze(0)  # [1, action_dim]
            
            # Predizer próximo estado e recompensa usando o RSSM
            state, reward = rssm.predict(state, action)
            total_reward += reward.item()
            
        total_rewards[j] = total_reward

        """
        
        for j in range(J):
            # Fazer cópia do ambiente para simular sem afetar o estado real
            temp_env = gym.make("MountainCarContinuous-v0") #TODO: trocar pelo rssm
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
            mu, logvar = encoder.forward(next_obs/255.0)
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

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                mu, logvar = encoder.forward(b_obs[mb_inds]/255.0)
                next_obs_latent = reparameterize(mu, logvar)
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(next_obs_latent, b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                #recon loss
                recon_images = decoder.forward(next_obs_latent)
                recon_loss = vae_loss(recon_images, b_obs[mb_inds] / 255, mu, logvar)
                recon_loss = recon_loss.mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef + recon_loss * args.recon_coef

                optimizer.zero_grad()
                encoder_optimizer.zero_grad()
                decoder_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                nn.utils.clip_grad_norm_(encoder.parameters(), args.max_grad_norm)
                nn.utils.clip_grad_norm_(decoder.parameters(), args.max_grad_norm)
                optimizer.step()
                encoder_optimizer.step()
                decoder_optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

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