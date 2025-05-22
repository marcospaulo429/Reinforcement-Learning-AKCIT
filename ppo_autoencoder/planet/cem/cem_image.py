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

from stable_baselines3.common.atari_wrappers import (  # isort:skip
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

import argparse
import heapq
import os
import time
from collections import defaultdict
from functools import partial
from multiprocessing import Pool

import gymnasium as gym
import numpy as np

from cem_auxiliar import setup_env, plot_history, setup_policy

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

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)

    if not os.path.exists(directory):
        os.makedirs(directory)


def get_elite_indicies(num_elite, rewards):
    return heapq.nlargest(num_elite, range(len(rewards)), key=rewards.__getitem__)


def evaluate_theta(theta, env_id, encoder, agent, device, monitor=False):
    """
    Avalia uma política (representada por seus parâmetros theta) em um ambiente,
    usando um encoder para processar a observação (assumida como imagem).

    Args:
        theta (np.ndarray): Os parâmetros da política.
        env_id (str): O ID do ambiente do Gymnasium a ser usado.
        encoder (torch.nn.Module): A rede encoder para processar as observações.
        agent (torch.nn.Module): O agente (contendo a política) que usa a representação latente.
        device (torch.device): O dispositivo Torch a ser usado (e.g., 'cpu' ou 'cuda').
        monitor (bool, optional): Se True, o ambiente será configurado para gravar um vídeo. Defaults to False.

    Returns:
        float: A recompensa total obtida pela política durante um episódio.
    """
    env, _, _ = setup_env(env_id, capture_video=monitor, run_name=env_id if monitor else "eval")
    policy = setup_policy(env, theta) # Você ainda pode usar uma política linear com a representação latente

    terminated = False
    truncated = False
    observation, _ = env.reset()
    episode_rewards = []

    # Mova a observação inicial para o dispositivo Torch
    obs = torch.Tensor(np.array([observation])).unsqueeze(0).to(device) / 255.0 # Normalize se a entrada do encoder for [0, 255]

    while not terminated and not truncated:
        with torch.no_grad():
            # Passa a observação pelo encoder para obter a representação latente
            mu, logvar = encoder(obs)            
            obs_latent = reparameterize(mu, logvar)

            # Obtém a ação do agente usando a representação latente
            # Adapte esta parte de acordo com a saída do seu 'agent'
            action = policy.act(observation)

        next_observation, reward, terminated, truncated, _ = env.step(action)
        episode_rewards.append(reward)

        # Prepara a próxima observação para a próxima iteração
        obs = torch.Tensor(np.array([next_observation])).unsqueeze(0).to(device) / 255.0

    env.close()
    return sum(episode_rewards)


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

    #device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    device = torch.device("mps" if torch.backends.mps.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    encoder = Encoder(args.latent_dim).to(device)
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=args.learning_rate, eps=1e-5)

    #CEM

    ensure_dir("./{}/".format(args.env_id))
    ensure_dir("./{}/videos/".format(args.env_id))

    start = time.time()
    num_episodes = args.epochs * args.num_process * args.batch_size
    print("expt of {} total episodes".format(num_episodes))

    num_elite = int(args.batch_size * args.elite_frac)
    history = defaultdict(list)

    env, _, act_shape = setup_env(args.env_id)
    obs_shape = args.latent_dim
    theta_dim = (obs_shape) * act_shape #TODO: ver se colocamos o 1 da obs_shape
    means = np.random.uniform(size=theta_dim)
    stds = np.ones(theta_dim)

    for epoch in range(args.epochs):
        extra_cov = max(1.0 - epoch / args.extra_decay_time, 0) * args.extra_std**2

        thetas = np.random.multivariate_normal(
            mean=means, cov=np.diag(np.array(stds**2) + extra_cov), size=args.batch_size
        )

        with Pool(args.num_process) as p:
            rewards = p.map(partial(evaluate_theta, env_id=args.env_id), thetas)

        rewards = np.array(rewards)

        indicies = get_elite_indicies(num_elite, rewards)
        elites = thetas[indicies]

        means = elites.mean(axis=0)
        stds = elites.std(axis=0)

        history["epoch"].append(epoch)
        history["avg_rew"].append(np.mean(rewards))
        history["std_rew"].append(np.std(rewards))
        history["avg_elites"].append(np.mean(rewards[indicies]))
        history["std_elites"].append(np.std(rewards[indicies]))

        print(
            "epoch {} - {:2.1f} {:2.1f} pop - {:2.1f} {:2.1f} elites".format(
                epoch,
                history["avg_rew"][-1],
                history["std_rew"][-1],
                history["avg_elites"][-1],
                history["std_elites"][-1],
            )
        )

    end = time.time()
    expt_time = end - start
    print("expt took {:2.1f} seconds".format(expt_time))

    plot_history(history, args.env_id, num_episodes, expt_time)
    num_optimal = 3
    print("epochs done - evaluating {} best thetas".format(num_optimal))

    best_theta_rewards = [
        evaluate_theta(theta, args.env_id, monitor=True) for theta in elites[:num_optimal]
    ]
    print("best rewards - {} across {} samples".format(best_theta_rewards, num_optimal))


    #############

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

    envs.close()
    writer.close()