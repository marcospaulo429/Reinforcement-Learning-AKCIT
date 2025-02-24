import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from dm_control import suite
from dm_control.suite.wrappers import pixels

from replay_buffer import ReplayBuffer
from world_model import (DreamerWorldModel, converter_cinza, 
                         get_data_loaders_from_replay_buffer, ver_reconstrucoes, 
                         collect_replay_buffer, train_world_model)
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
    num_iterations = 3
    update_step = 1
    repositorio = "dreamer/model_4"
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Usando device:", device)
    
    writer = SummaryWriter(log_dir=repositorio)
    
    env = suite.load(domain_name="cartpole", task_name="balance")
    env = pixels.Wrapper(env, pixels_only=True,
                         render_kwargs={'height': HEIGHT, 'width': WIDTH, 'camera_id': 0})
    
    replay_buffer = ReplayBuffer()
    replay_buffer = collect_replay_buffer(env, S, replay_buffer)
    
    action_dim = env.action_spec().shape[0]
    
    # Carrega o World Model treinado
    world_model = DreamerWorldModel(input_size, latent_dim, action_dim, hidden_dim).to(device)
    world_model.load_state_dict(torch.load("world_model/model_6/world_model_weights.pth"))
    wm_optimizer = optim.Adam(world_model.parameters(), lr=0.001)
    mse_loss = nn.MSELoss()
    
    # Inicializa os modelos de comportamento
    actor = Actor(latent_dim, action_dim).to(device)
    value_net = ValueNet(latent_dim).to(device)
    actor_optimizer = optim.Adam(actor.parameters(), lr=1e-3)
    value_optimizer = optim.Adam(value_net.parameters(), lr=1e-3)
    
    rewards_history = []
    
    for iteration in range(num_iterations):
        # Obtém os data loaders a partir do replay buffer
        train_loader, test_loader = get_data_loaders_from_replay_buffer(
            replay_buffer, batch_size=batch_size, HEIGHT=HEIGHT, WIDTH=WIDTH)
        
        for it in range(update_step):
            print(f"\nUpdate Step {it+1}/{update_step} da iteração {iteration+1}/{num_iterations}")
            
            # Treinamento do World Model
            epochs_wm = 3
            loss_train_history, loss_test_history, reward_train_history, reward_test_history = train_world_model(
                epochs_wm, world_model, train_loader, test_loader, device, hidden_dim, mse_loss, wm_optimizer, writer)
            
            # Registra as métricas do world model no TensorBoard
            for epoch in range(epochs_wm):
                global_epoch = iteration * epochs_wm + epoch
                writer.add_scalar("WorldModel/TrainLoss", loss_train_history[epoch], global_epoch)
                writer.add_scalar("WorldModel/TestLoss", loss_test_history[epoch], global_epoch)
                writer.add_scalar("WorldModel/RewardTrainLoss", reward_train_history[epoch], global_epoch)
                writer.add_scalar("WorldModel/RewardTestLoss", reward_test_history[epoch], global_epoch)
            
            # Treinamento do comportamento (Actor e ValueNet)
            epochs_behavior = 3
            actor, value_net, actor_loss_history, value_loss_history = behavior_learning(
                world_model, actor, value_net, epochs_behavior, train_loader, device, writer,
                horizon=5, gamma=0.99,
                value_optimizer=value_optimizer, actor_optimizer=actor_optimizer,
                mse_loss=mse_loss)
            
            # Registra as métricas do comportamento no TensorBoard
            for epoch in range(epochs_behavior):
                global_epoch = iteration * epochs_behavior + epoch
                writer.add_scalar("Behavior/ActorLoss", actor_loss_history[epoch], global_epoch)
                writer.add_scalar("Behavior/ValueLoss", value_loss_history[epoch], global_epoch)
        
        # INTERAÇÃO COM O AMBIENTE
        print(f"Coletando interação {iteration+1} no ambiente")
        time_step = env.reset()
        done = False
        obs_atual = converter_cinza(time_step.observation['pixels'])
        obs_atual = obs_atual.astype(np.float32) / 127.5 - 1.0

        episode_data = []
        steps_done = 0
        steps = 500
        ep_rewards = 0

        while steps_done < steps:
            # Converte a observação para tensor com shape (1, 1, 84, 84)
            obs_tensor = torch.tensor(obs_atual).view(1, 1, 84, 84).to(device)
            conv_out = world_model.autoencoder.encoder_conv(obs_tensor)  # (1, 64, 21, 21)
            conv_out = conv_out.view(conv_out.size(0), -1)               # (1, 28224)
            latent = world_model.autoencoder.encoder_fc(conv_out)        # (1, latent_dim)
            
            action_tensor = actor(latent)
            action_np = action_tensor.detach().cpu().numpy()[0]
            
            # TODO: Estratégia de exploração
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
        print(f"Recompensa acumulada = {ep_rewards:.2f}")
        writer.add_scalar("Reward/Episode", ep_rewards, iteration)
        
        # Registra um histograma da distribuição do vetor latente para o primeiro batch
        for batch in test_loader:
            obs, _, _, _ = next(iter(train_loader))
            if obs.dim() == 2 and obs.size(1) == 84 * 84:
                obs = obs.view(obs.size(0), 1, 84, 84)
            obs = obs.to(device)
            conv_out = world_model.autoencoder.encoder_conv(obs)
            conv_out = conv_out.view(conv_out.size(0), -1)
            latent = world_model.autoencoder.encoder_fc(conv_out)
            writer.add_histogram("Latent/Distribution", latent, iteration)
            break  
    
    print("\nTreinamento finalizado para essa fase!")
    ver_reconstrucoes(world_model, test_loader, device, input_size, num_samples=8,
                      action_dim=action_dim, hidden_dim=hidden_dim, HEIGHT=HEIGHT, WIDTH=WIDTH)
    
    with open(f"{repositorio}/training_details.txt", "a") as log_file:  
        log_file.write(f"Height: {HEIGHT}, Width: {WIDTH}, Hidden Dimension: {hidden_dim}, "
                       f"Latent Dimension: {latent_dim}, Batch size: {batch_size}, "
                       f"Iterations: {num_iterations}, Update iterations: {update_step}\n")
    
    writer.close()

if __name__ == "__main__":
    main()
