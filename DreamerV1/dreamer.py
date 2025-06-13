# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_ataripy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from models import TransitionModel, RewardModel
from utils import test_world_model
from dynamics_learning import dynamics_learning
from behavior_learning import imagine_trajectories, behavior_learning

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
    treshold_save_model_reward: float = 100.0

    belief_size: int = 200 # Tamanho do estado recorrente (h)
    hidden_size: int = 200 # Tamanho das camadas ocultas nas MLPs do World Model
    future_rnn: bool = True # Se a GRU do transition model deve ser usada para o post_rnn_layers
    # action_dim: int = ? # Será obtido de envs.single_action_space.n
    mean_only: bool = False # Se deve amostrar apenas a média ou usar reparameterização
    min_stddev: float = 0.1 # Desvio padrão mínimo para a distribuição
    num_layers: int = 2 # Número de camadas nas MLPs do World Model

    kl_beta: float = 1.0 # Peso para a perda KL do World Model
    reward_beta: float = 1.0 # Peso para a perda de recompensa do World Model
    horizon_to_imagine: int = 10

    # Algorithm specific arguments
    env_id: str = "BreakoutNoFrameskip-v4"
    """the id of the environment"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 14
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
    lambda_return_gamma: float = 0.99
    """The discount factor gamma for lambda-returns."""
    lambda_return_lambda: float = 0.95
    """The lambda value for lambda-returns."""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""
    weights_saved: bool = False
    best_episodic_return: float = 0.0


def make_env(env_id, idx, capture_video, run_name):
    def thunk():
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

    if args.cuda:
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

    agent = Agent(args.latent_dim, envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    encoder = Encoder(args.latent_dim).to(device)
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=args.learning_rate)

    decoder = Decoder(args.latent_dim).to(device)   
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=1e-4)

    args.action_dim = envs.single_action_space.n 
    transition_model = TransitionModel(
        args.latent_dim, 
        args.belief_size,
        args.hidden_size,
        args.future_rnn,
        args.action_dim,
        args.mean_only,
        args.min_stddev,
        args.num_layers
    ).to(device)
    transition_optimizer = optim.Adam(transition_model.parameters(), lr=1e-4)

    # RewardModel usa belief_size e latent_dim (para sample)
    reward_model = RewardModel(
        args.hidden_size, # hidden_dim para o RewardModel (geralmente pode ser belief_size ou hidden_size do WM)
        args.latent_dim   # state_dim para o RewardModel (o sample do estado latente)
    ).to(device)
    reward_optimizer = optim.Adam(reward_model.parameters(), lr=1e-3)

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
                        episodic_return = info["episode"]["r"]
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
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape) #(num_envs* num_steps, channels, altura, largura)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)  #(num_envs* num_steps)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1) 
        print(obs.shape, b_obs.shape, b_actions.shape)

        #data for wm
        obs_seq = obs.clone() #(steps, num_envs, channels, altura, altura)
        actions_seq = actions.clone().long() #(steps, num_envs)
        rewards_seq = rewards.clone() #(steps, num_envs)
        dones_seq = dones.clone() #(steps, num_envs)

        #DYNAMICS LEARNING
        total_loss, kl_loss, reward_loss, recon_loss, obs_latents_wm_tomodel = dynamics_learning(
            args, transition_model, reward_model, encoder, decoder, 
            transition_optimizer, reward_optimizer, encoder_optimizer, 
            decoder_optimizer, obs_seq, actions_seq, rewards_seq, device, vae_loss
        )

        # --- BEHAVIOR LEARNING: Otimização do Ator (Política) e Crítico (Valor) ---
        
        actor_losses, critic_losses, total_behavior_losses, entropies, clipfracs, approx_kls = behavior_learning(
            args, 
            obs_latents_wm_tomodel, 
            actions_seq,
            encoder, 
            transition_model, 
            reward_model, 
            agent, 
            optimizer, 
            imagine_trajectories, 
            device
        )
        break

        # --- Métricas para logging do Behavior Learning ---
        writer.add_scalar("losses/behavior_learning/actor_loss", actor_loss.item(), global_step)
        writer.add_scalar("losses/behavior_learning/critic_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/behavior_learning/entropy_loss", entropy.item(), global_step)
        # Se você mantiver as métricas PPO-like
        writer.add_scalar("losses/behavior_learning/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/behavior_learning/clipfrac", np.mean(clipfracs), global_step)

        # Calcula explained_variance para o Behavior Learning
        with torch.no_grad():
            # Use todos os estados imaginados (imagined_states) para obter a previsão final do crítico
            final_critic_predictions = agent.critic(imagined_states).view(-1).cpu().numpy()
            # Use os lambda_returns calculados
            true_lambda_returns_np = lambda_returns_flat.cpu().numpy()
            
            var_y_behavior = np.var(true_lambda_returns_np)
            explained_var_behavior = np.nan if var_y_behavior == 0 else 1 - np.var(true_lambda_returns_np - final_critic_predictions) / var_y_behavior
            writer.add_scalar("losses/behavior_learning/explained_variance", explained_var_behavior, global_step)


        # TRY NOT TO MODIFY: record rewards for plotting purposes
        

        # --- Métricas do World Model ---
        writer.add_scalar("losses/world_model/kl_loss", kl_loss_wm.item(), global_step)
        writer.add_scalar("losses/world_model/reward_loss", loss_reward_wm.item(), global_step)
        writer.add_scalar("losses/world_model/total_loss", total_world_model_loss.item(), global_step)
        
        # Opcional: Aprender a taxa de aprendizado dos otimizadores do World Model
        writer.add_scalar("charts/learning_rate_wm_transition", transition_optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("charts/learning_rate_wm_reward", reward_optimizer.param_groups[0]["lr"], global_step)

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

        # --- Seção para gravar os pesos do modelo ---
        if episodic_return >= args.treshold_save_model_reward and (not args.weights_saved or (episodic_return > 1.08 * args.best_episodic_return)):
            args.best_episodic_return = episodic_return
            print(f"Recompensa média ({episodic_return}) atingiu o limiar de {args.treshold_save_model_reward}. Salvando pesos...")
            
            # Cria o diretório para salvar os modelos, se não existir
            model_save_path = f"models/{run_name}"
            os.makedirs(model_save_path, exist_ok=True)

            # Salva os state_dicts de todos os modelos
            torch.save(agent.state_dict(), os.path.join(model_save_path, f"agent_global_step_{global_step}.pt"))
            torch.save(encoder.state_dict(), os.path.join(model_save_path, f"encoder_global_step_{global_step}.pt"))
            torch.save(decoder.state_dict(), os.path.join(model_save_path, f"decoder_global_step_{global_step}.pt"))
            torch.save(transition_model.state_dict(), os.path.join(model_save_path, f"transition_model_global_step_{global_step}.pt"))
            torch.save(reward_model.state_dict(), os.path.join(model_save_path, f"reward_model_global_step_{global_step}.pt"))
            print(f"Pesos dos modelos salvos em: {model_save_path}")
            weights_saved = True 


        concatenated_images_tensor, imagined_rewards = test_world_model(
            global_step,
            writer,
            encoder,
            decoder,
            transition_model,
            reward_model,
            obs_seq,
            actions_seq,
            args,
            device,
            reparameterize        
            )
        writer.add_image("world_model/imagined_trajectory_concatenated", concatenated_images_tensor, global_step)
        writer.add_scalar("world_model/imagined_total_reward_sum", imagined_rewards, global_step)

    envs.close()
    writer.close()