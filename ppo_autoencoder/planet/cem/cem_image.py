import argparse
import heapq
import os
import time
from collections import defaultdict
from functools import partial
from multiprocessing import Pool

from dataclasses import dataclass
import tyro

import gymnasium as gym
import numpy as np

import torch
import torch.nn as nn

from cem_auxiliar import setup_env, plot_history, setup_policy

@dataclass
class Args:
    env_id: str = "BreakoutNoFrameskip-v4"
    """the id of the Atari environment"""
    num_process: int = 2
    """number of parallel processes for evaluation"""
    epochs: int = 50
    """number of CEM iterations"""
    cem_batch_size: int = 256
    """number of samples (thetas) per CEM epoch"""
    elite_frac: float = 0.1
    """fraction of elites to select from the batch"""
    extra_std: float = 0.5
    """initial extra standard deviation for exploration noise"""
    extra_decay_time: float = 25.0
    """time (in epochs) for extra_std to decay"""
    run_name: str = "cem_atari_run"
    """name for this run (used for video folder structure if setup_env uses it)"""
    latent_dim: int = 256
    """latent dimension of the encoder output"""
    cuda: bool = True
    """if toggled, attempt to use CUDA or MPS (Apple Silicon GPU)"""


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def get_elite_indicies(num_elite, rewards):
    return heapq.nlargest(num_elite, range(len(rewards)), key=rewards.__getitem__)

# --- Funções e Classe do Encoder ---
def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Encoder(nn.Module):
    def __init__(self, latent_dim=256):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim

        self.encoder_cnn = nn.Sequential(
            layer_init(nn.Conv2d(4, 16, kernel_size=8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(16, 32, kernel_size=4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 32, kernel_size=3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        self.fc_mu = layer_init(nn.Linear(32 * 7 * 7, latent_dim))
        self.fc_logvar = layer_init(nn.Linear(32 * 7 * 7, latent_dim))

    def forward(self, x):
        x = self.encoder_cnn(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
# --- Fim das Funções e Classe do Encoder ---


def evaluate_theta(theta_params_np, env_id_str, encoder_state_dict, latent_dim, monitor_video_flag=False):
    # Determine o dispositivo para este processo
    if torch.cuda.is_available():
        eval_device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        eval_device = torch.device("mps")
    else:
        eval_device = torch.device("cpu")
    
    # Crie uma NOVA instância do encoder neste processo filho
    current_encoder_instance = Encoder(latent_dim=latent_dim).to(eval_device)
    # Carregue os pesos que foram passados como state_dict
    current_encoder_instance.load_state_dict(encoder_state_dict)
    current_encoder_instance.eval() # Coloque em modo de avaliação (sem dropout, batchnorm etc.)

    env, _, _ = setup_env(env_id_str)

    if monitor_video_flag:
        video_folder = f"./{env_id_str}/videos/cem_monitor/"
        ensure_dir(video_folder)
        env = gym.wrappers.RecordVideo(env, video_folder=video_folder, name_prefix="cem_eval")

    policy = setup_policy(env, theta_params_np, use_latent=True, latent_dim=latent_dim)

    terminated = False
    truncated = False
    observation_np, _ = env.reset()

    total_rewards = 0.0

    while not terminated and not truncated:
        with torch.no_grad():
            obs_tensor = torch.from_numpy(np.array(observation_np)).float().unsqueeze(0).to(eval_device)
            obs_tensor_normalized = obs_tensor / 255.0
            
            mu, logvar = current_encoder_instance(obs_tensor_normalized)
            latent_representation = reparameterize(mu, logvar)
        
        action = policy.act(latent_representation.cpu().numpy().squeeze(0)) 

        next_observation_np, reward, terminated, truncated, _ = env.step(action)

        total_rewards += reward
        observation_np = next_observation_np

    env.close()
    return total_rewards


def run_cem(args: Args):
    
    if args.cuda:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
            print("Warning: CUDA/MPS specified but not available. Falling back to CPU.")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    ensure_dir(f"./{args.env_id}/")
    ensure_dir(f"./{args.env_id}/videos/")

    start_time = time.time()
    num_episodes = args.epochs * args.cem_batch_size
    print(f"expt of {num_episodes} total episodes")

    num_elite = int(args.cem_batch_size * args.elite_frac)
    if num_elite == 0 and args.cem_batch_size > 0: num_elite = 1
    history = defaultdict(list)

    encoder = Encoder(latent_dim=args.latent_dim).to(device)
    
    encoder.cpu() # Mova para CPU para serialização
    encoder_state_dict = encoder.state_dict()
    encoder.to(device) # Mova de volta para o dispositivo principal para uso posterior, se houver


    temp_env, _, act_shape = setup_env(args.env_id)
    temp_env.close()
    print(f"Environment: {args.env_id}, Action Dim: {act_shape}, Encoder Latent Dim: {args.latent_dim}")

    theta_dim = args.latent_dim * act_shape + act_shape # (latent_dim * num_actions) + num_actions (bias)
    means = np.random.uniform(size=theta_dim)
    stds = np.ones(theta_dim) * 0.5

    evaluate_theta_partial = partial(evaluate_theta,
                                     env_id_str=args.env_id,
                                     encoder_state_dict=encoder_state_dict,
                                     latent_dim=args.latent_dim,
                                     monitor_video_flag=False)

    for epoch in range(args.epochs):
        extra_cov = max(1.0 - epoch / args.extra_decay_time, 0) * args.extra_std**2

        thetas = np.random.multivariate_normal(
            mean=means, cov=np.diag(np.array(stds**2) + extra_cov), size=args.cem_batch_size
        )
        
        # CORREÇÃO: Não converter 'thetas' para lista.
        # 'thetas' já é um np.ndarray e pode ser mapeado diretamente.
        # Removido: map_args = thetas.tolist()
        map_args = thetas # Basta usar o array numpy diretamente

        if args.num_process > 1:
            with Pool(args.num_process) as p:
                rewards = p.map(evaluate_theta_partial, map_args)
        else:
            # Para execução sequencial, a entrada para evaluate_theta_partial
            # deve ser um np.ndarray individual, não uma lista.
            rewards = [evaluate_theta_partial(theta_params_np=theta) for theta in map_args]

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

    end_time = time.time()
    expt_time = end_time - start_time
    print(f"expt took {expt_time:.1f} seconds")

    plot_history(history, args.env_id, num_episodes, expt_time)
    num_optimal = 3
    print(f"epochs done - evaluating {num_optimal} best thetas")

    best_theta_rewards = []
    for i, elite_theta_np in enumerate(elites[:num_optimal]):
        reward = evaluate_theta(theta_params_np=elite_theta_np,
                                env_id_str=args.env_id,
                                encoder_state_dict=encoder_state_dict,
                                latent_dim=args.latent_dim,
                                monitor_video_flag=True)
        best_theta_rewards.append(reward)
        print(f"Monitored Elite {i+1}/{num_optimal} Reward: {reward:.1f}")

    print(f"best rewards - {best_theta_rewards} across {num_optimal} samples")


if __name__ == "__main__":
    args = tyro.cli(Args)
    print("--- Starting CEM with Config ---")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("-------------------------------")
    run_cem(args)