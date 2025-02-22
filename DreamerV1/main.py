import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from dm_control import suite
from dm_control.suite.wrappers import pixels

from replay_buffer import ReplayBuffer
from world_model import DreamerWorldModel, converter_cinza, get_data_loaders_from_replay_buffer, ver_reconstrucoes, collect_replay_buffer, train_world_model
from behavior_learning import Actor, ValueNet, behavior_learning
from torch.utils.tensorboard import SummaryWriter

def main():
    HEIGHT = 84
    WIDTH = 84
    hidden_dim = 256
    input_size = HEIGHT * WIDTH
    latent_dim = 256
    batch_size = 32
    S = 5
    num_iterations = 15
    update_step = 1
    repositorio = "dreamer/model_3"
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    print("Usando device:", device)
    
    env = suite.load(domain_name="cartpole", task_name="balance")
    env = pixels.Wrapper(env, pixels_only=True,
                         render_kwargs={'height': HEIGHT, 'width': WIDTH, 'camera_id': 0})
    
    replay_buffer = ReplayBuffer()
    replay_buffer = collect_replay_buffer(env, S, replay_buffer)
    
    action_dim = env.action_spec().shape[0]
    
    world_model = DreamerWorldModel(input_size, latent_dim, action_dim, hidden_dim).to(device)
    world_model.load_state_dict(torch.load("world_model/model_3/world_model_weights.pth"))
    wm_optimizer = optim.Adam(world_model.parameters(), lr=0.001)
    mse_loss = nn.MSELoss()
    
    actor = Actor(latent_dim, action_dim).to(device)
    value_net = ValueNet(latent_dim).to(device)
    actor_optimizer = optim.Adam(actor.parameters(), lr=1e-3)
    value_optimizer = optim.Adam(value_net.parameters(), lr=1e-3)
    
    writer = SummaryWriter(f"{repositorio}")
    
    rewards_history = []
    
    for iteration in range(num_iterations):
        train_loader, test_loader = get_data_loaders_from_replay_buffer(replay_buffer, batch_size=batch_size, HEIGHT=HEIGHT, WIDTH=WIDTH)
        for it in range(update_step):
            print(f"\n Update Step {it+1}/{num_iterations} (Total Update Step: {update_step})")
            
            epochs_wm = 3
            train_world_model(epochs_wm, world_model, train_loader, test_loader, device, hidden_dim, mse_loss, wm_optimizer, writer)
            
            epochs_behavior = 3
            actor, value_net = behavior_learning(world_model, actor, value_net, epochs_behavior, train_loader, device, writer,
                                                  horizon=5, gamma=0.99,
                                                  value_optimizer=value_optimizer, actor_optimizer=actor_optimizer,
                                                  mse_loss=mse_loss)
            
        # ENVIRONMENT INTERACTION
        print(f"Coletando passo {iteration+1} no environment")
        time_step = env.reset()
        done = False
        obs_atual = converter_cinza(time_step.observation['pixels'])
        obs_atual = obs_atual.astype(np.float32) / 127.5 - 1.0

        episode_data = []
        steps_done = 0
        steps=500
        ep_rewards=0

        while steps_done < steps:
            
            # Converte a observação para tensor com shape (1, 1, 84, 84)
            obs_tensor = torch.tensor(obs_atual).view(1, 1, 84, 84).to(device)
            conv_out = world_model.autoencoder.encoder_conv(obs_tensor)  # (1, 64, 21, 21)
            conv_out = conv_out.view(conv_out.size(0), -1)               # (1, 28224)
            latent = world_model.autoencoder.encoder_fc(conv_out)        # (1, latent_dim)
            
            action_tensor = actor(latent)
            action_np = action_tensor.detach().cpu().numpy()[0]
            
            # TODO Estratégia de exploração

            time_step = env.step(action_np)
            done = time_step.last()
            reward = time_step.reward if time_step.reward is not None else 0.0
            ep_rewards += reward

            obs_prox = converter_cinza(time_step.observation['pixels'])
            obs_prox = obs_prox.astype(np.float32) / 127.5 - 1.0

            step_data = {
                "obs": obs_atual,
                "action": action_np,
                "reward": reward,
                "next_obs": obs_prox,
                "done": done
            }
            episode_data.append(step_data)
            obs_atual = obs_prox
            steps_done += 1

            if done:
                replay_buffer.add_episode(episode_data)
                episode_data = []
                time_step = env.reset()
                done = False
                obs_atual = converter_cinza(time_step.observation['pixels'])
                obs_atual = obs_atual.astype(np.float32) / 127.5 - 1.0
            
        rewards_history.append(ep_rewards)
        print(f"Recompensa  = {ep_rewards:.2f}")
        writer.add_scalar("Reward", ep_rewards, iteration)
        
        for batch in test_loader:
            obs, _, _, _ = next(iter(train_loader))

            if obs.dim() == 2 and obs.size(1) == 84*84:
                obs = obs.view(obs.size(0), 1, 84, 84)

            obs = obs.to(device)
            conv_out = world_model.autoencoder.encoder_conv(obs)  # Saída: (B, 64, 21, 21)
            conv_out = conv_out.view(conv_out.size(0), -1)           # (B, 28224)
            latent = world_model.autoencoder.encoder_fc(conv_out)    # (B, latent_dim)

            writer.add_histogram("Latent/Distribution", latent, iteration)
            break  
    
    print("\nTreinamento finalizado para essa fase!")
    ver_reconstrucoes(world_model, test_loader, device, input_size, num_samples=8,
                        action_dim=action_dim, hidden_dim=hidden_dim, HEIGHT=HEIGHT, WIDTH=WIDTH)
    
    
    with open(f"{repositorio}/training_details.txt", "a") as log_file:  
        log_file.write(f"Height: {HEIGHT}, Width: {WIDTH}, Hiddem Dimension: {hidden_dim}, latent dimension: {latent_dim}, Batch size: {batch_size}, Iterations: {num_iterations}, Update iterations of the models: {update_step}\n")
    
if __name__ == "__main__":
    main()
