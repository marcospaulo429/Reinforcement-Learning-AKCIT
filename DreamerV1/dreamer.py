# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_ataripy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

from models import TransitionModel, RewardModel, Encoder, Decoder, Agent, reparameterize, vae_loss
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
    latent_dim: int = 64
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL_dreamer"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    treshold_save_model_reward: float = 100.0

    belief_size: int = latent_dim # Tamanho do estado recorrente (h)
    hidden_size: int = 200 # Tamanho das camadas ocultas nas MLPs do World Model
    future_rnn: bool = True # Se a GRU do transition model deve ser usada para o post_rnn_layers
    # action_dim: int = ? # Será obtido de envs.single_action_space.n
    mean_only: bool = False # Se deve amostrar apenas a média ou usar reparameterização
    min_stddev: float = 0.1 # Desvio padrão mínimo para a distribuição
    num_layers: int = 2 # Número de camadas nas MLPs do World Model

    kl_beta: float = 1.0 # Peso para a perda KL do World Model
    reward_beta: float = 1.0 # Peso para a perda de recompensa do World Model
    horizon_to_imagine: int = 15

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

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape) #(num_envs* num_steps, channels, altura, largura)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)  
        b_values = values.reshape(-1) 
        print(obs.shape, b_obs.shape, b_actions.shape)

        #data for wm
        obs_seq = obs.clone() #(steps, num_envs, channels, altura, altura)
        actions_seq = actions.clone().long() #(steps, num_envs)
        rewards_seq = rewards.clone() #(steps, num_envs)
        dones_seq = dones.clone() #(steps, num_envs)

        #DYNAMICS LEARNING
        encoder, decoder, transition_model, reward_model, total_wm_loss, kl_loss_wm, reward_loss, recon_loss, obs_latents_wm_tomodel = dynamics_learning(
            args, transition_model, reward_model, encoder, decoder, 
            transition_optimizer, reward_optimizer, encoder_optimizer, 
            decoder_optimizer, obs_seq, actions_seq, rewards_seq, device, vae_loss
        )

        # --- BEHAVIOR LEARNING: Otimização do Ator (Política) e Crítico (Valor) ---
        agent, actor_losses_list, critic_losses_list, total_behavior_losses_list, \
entropies_list, clipfracs_list, approx_kls_list  = behavior_learning(args, obs_latents_wm_tomodel, agent,
                                                                     optimizer, imagine_trajectories, device, transition_model, reward_model)
        
        
        writer.add_scalar("losses/behavior_learning/actor_loss", np.mean(actor_losses_list), global_step)
        writer.add_scalar("losses/behavior_learning/critic_loss", np.mean(critic_losses_list), global_step)
        writer.add_scalar("losses/behavior_learning/entropy", np.mean(entropies_list), global_step) 
        writer.add_scalar("losses/behavior_learning/approx_kl", np.mean(approx_kls_list), global_step)
        writer.add_scalar("losses/behavior_learning/clipfrac", np.mean(clipfracs_list), global_step)

        writer.add_scalar("losses/world_model/kl_loss", kl_loss_wm.item(), global_step)
        writer.add_scalar("losses/world_model/reconstruction_loss", recon_loss.item(), global_step) 
        writer.add_scalar("losses/world_model/reward_loss", reward_loss.item(), global_step)
        writer.add_scalar("losses/world_model/total_loss", total_wm_loss.item(), global_step)

        writer.add_scalar("charts/learning_rate_wm_transition", transition_optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("charts/learning_rate_wm_reward", reward_optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("charts/learning_rate_wm_encoder", encoder_optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("charts/learning_rate_wm_decoder", decoder_optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("charts/learning_rate_agent", optimizer.param_groups[0]["lr"], global_step)
    

        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)


        with torch.no_grad():
            sample_obs = b_obs[:10].to(device) / 255.0
            mu, logvar = encoder(sample_obs)
            z = reparameterize(mu, logvar)
            recon = decoder(z)

            if sample_obs.shape[1] == 1:
                sample_obs_rgb = sample_obs.repeat(1, 3, 1, 1)
                recon_rgb = recon.repeat(1, 3, 1, 1)
            elif sample_obs.shape[1] == 3:
                sample_obs_rgb = sample_obs
                recon_rgb = recon
            else:
                # Fallback para outros números de canais, pegando os 3 primeiros ou avisando
                print(f"Atenção: Número de canais de observação ({sample_obs.shape[1]}) não é 1 nem 3. Ajuste a visualização.")
                sample_obs_rgb = sample_obs[:, :3, :, :] if sample_obs.shape[1] >= 3 else sample_obs
                recon_rgb = recon[:, :3, :, :] if recon.shape[1] >= 3 else recon

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