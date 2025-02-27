import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader

from dm_control import suite
from dm_control.suite.wrappers import pixels

from auxiliares import training_device
from replay_buffer import ReplayBuffer, sample_data_sequences, converter_cinza
from world_model import (DreamerWorldModel, 
                         get_data_loaders_from_replay_buffer, 
                         collect_replay_buffer, train_world_model)
from behavior_learning import behavior_learning, extract_latent_sequences, create_latent_dataset
from actor_critic import ActionModel, ValueNet

import wandb

# Variável global para o contador de steps
global_step = 0

def log_wandb(metrics):
    """Loga métricas no wandb usando o contador global e o incrementa."""
    global global_step
    wandb.log(metrics, step=global_step)
    global_step += 1

# --- Função de checkpoint ---
def load_checkpoint(checkpoint_path, world_model, actor, value_net,
                    wm_optimizer, actor_optimizer, value_optimizer):
    """
    Carrega os pesos e estados dos otimizadores a partir do checkpoint salvo.
    
    Retorna:
        iteration (int): A iteração a partir da qual continuar o treinamento.
    """
    if os.path.exists(checkpoint_path):
        print(f"Carregando checkpoint de {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        world_model.load_state_dict(checkpoint["world_model_state"])
        actor.load_state_dict(checkpoint["actor_state"])
        value_net.load_state_dict(checkpoint["value_net_state"])
        wm_optimizer.load_state_dict(checkpoint["optimizer_world_model"])
        actor_optimizer.load_state_dict(checkpoint["optimizer_actor"])
        value_optimizer.load_state_dict(checkpoint["optimizer_value_net"])
        iteration = checkpoint.get("iteration", 0)
        print(f"Checkpoint carregado. Retomando a partir da iteração {iteration}")
        return iteration
    else:
        print("Nenhum checkpoint encontrado. Iniciando treinamento do zero.")
        return 0

# --- Função main ---
def main():
    HEIGHT = 84
    WIDTH = 84
    hidden_dim = 256
    input_size = HEIGHT * WIDTH
    latent_dim = 256
    batch_size = 1250
    epochs_wm_behavior = 5
    num_iterations = 1000
    update_step = 1
    repositorio = "dreamer/model_15"
    device = training_device()
    print("Usando device:", device)
    
    env = suite.load(domain_name="cartpole", task_name="swingup")
    env = pixels.Wrapper(env, pixels_only=True,
                         render_kwargs={'height': HEIGHT, 'width': WIDTH, 'camera_id': 0})
    
    replay_buffer = ReplayBuffer()
    
    action_dim = env.action_spec().shape[0]
    
    # Carrega o World Model treinado
    world_model = DreamerWorldModel(input_size, latent_dim, action_dim, hidden_dim).to(device)
    world_model.load_state_dict(torch.load("dreamer/model_15/world_model_weights.pth"))
    wm_optimizer = optim.Adam(world_model.parameters(), lr=6e-4)
    mse_loss = nn.MSELoss()
    
    actor = ActionModel(latent_dim, action_dim).to(device)
    value_net = ValueNet(latent_dim).to(device)
    actor_optimizer = optim.Adam(actor.parameters(), lr=8e-5)
    value_optimizer = optim.Adam(value_net.parameters(), lr=8e-5)
    
    start_iteration = 0  # Inicia do zero, se não usar checkpoint

    checkpoint_path = os.path.join(repositorio, "checkpoint.pth")
    start_iteration = load_checkpoint(checkpoint_path, world_model, actor, value_net,
                                      wm_optimizer, actor_optimizer, value_optimizer)
    
    # Inicializa o wandb e configura os hiperparâmetros
    wandb.init(project="dreamer_14", config={
        "HEIGHT": HEIGHT,
        "WIDTH": WIDTH,
        "hidden_dim": hidden_dim,
        "latent_dim": latent_dim,
        "batch_size": batch_size,
        "epochs_wm_behavior": epochs_wm_behavior,
        "num_iterations": num_iterations,
        "update_step": update_step
    })
    
    wandb.define_metric("Reward/Episode", step_metric="global_step", summary="max")
    
    replay_buffer = collect_replay_buffer(env, 5, replay_buffer)
    
    
    rewards_history = []
    
    for iteration in range(start_iteration, num_iterations):
        # Coleta sequências de dados do replay buffer
        data_sequence = sample_data_sequences(replay_buffer, num_sequences=50, sequence_length=50)
        train_loader, test_loader = get_data_loaders_from_replay_buffer(
            data_sequence, batch_size=batch_size, HEIGHT=HEIGHT, WIDTH=WIDTH)
        
        for it in range(update_step):
            print(f"\nUpdate Step {it+1}/{update_step} da iteração {iteration+1}/{num_iterations}")
            
            
            if iteration % 15 == 0:
                # Treinamento do World Model
                loss_train_history, loss_test_history, reward_train_history, reward_test_history = train_world_model(
                    15, world_model, train_loader, test_loader, device, hidden_dim, mse_loss, wm_optimizer)
                
                # Registra as métricas do world model no wandb
                for epoch in range(epochs_wm_behavior):
                    log_wandb({
                        "WorldModel/TrainLoss": loss_train_history[epoch],
                        "WorldModel/TestLoss": loss_test_history[epoch],
                        "WorldModel/RewardTrainLoss": reward_train_history[epoch],
                        "WorldModel/RewardTestLoss": reward_test_history[epoch]
                    })
            
            # Treinamento do comportamento (Actor e ValueNet)
            epochs_behavior = epochs_wm_behavior
            latent_buffer = extract_latent_sequences(world_model, data_sequence, device)
            latent_dataset = create_latent_dataset(latent_buffer)
            latent_loader = DataLoader(latent_dataset, batch_size=batch_size)
            actor, value_net, actor_loss_history, value_loss_history = behavior_learning(
                world_model, actor, value_net,
                latent_loader=latent_loader,
                device=device,
                horizon=15,
                gamma=0.99,
                value_optimizer=value_optimizer,
                actor_optimizer=actor_optimizer,
                mse_loss=mse_loss,
                epochs_behavior=epochs_behavior
            )
            
            # Registra as métricas do comportamento no wandb
            for epoch in range(epochs_behavior):
                log_wandb({
                    "Behavior/ActorLoss": actor_loss_history[epoch],
                    "Behavior/ValueLoss": value_loss_history[epoch]
                })
        
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
            obs_tensor = torch.tensor(obs_atual).view(1, 1, 84, 84).to(device)
            conv_out = world_model.autoencoder.encoder_conv(obs_tensor)  # (1, C, H, W)
            conv_out = conv_out.view(conv_out.size(0), -1)
            latent = world_model.autoencoder.encoder_fc(conv_out)        # (1, latent_dim)
            
            dist, mean, std = actor(latent)
            action_tensor = dist.rsample() 
            action_np = action_tensor.detach().cpu().numpy()[0]
            
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
        log_wandb({"Reward/Episode": ep_rewards})
        
        # Registra um histograma da distribuição do vetor latente para o primeiro batch
        for batch in test_loader:
            obs, _, _, _ = next(iter(train_loader))
            if obs.dim() == 2 and obs.size(1) == 84 * 84:
                obs = obs.view(obs.size(0), 1, 84, 84)
            obs = obs.to(device)
            conv_out = world_model.autoencoder.encoder_conv(obs)
            conv_out = conv_out.view(conv_out.size(0), -1)
            latent = world_model.autoencoder.encoder_fc(conv_out)
            log_wandb({"Latent/Distribution": wandb.Histogram(latent.cpu().detach().numpy())})
            break  

        # Salva um checkpoint para retomar o treinamento futuramente
        checkpoint = {
            "iteration": iteration + 1,
            "world_model_state": world_model.state_dict(),
            "actor_state": actor.state_dict(),
            "value_net_state": value_net.state_dict(),
            "optimizer_world_model": wm_optimizer.state_dict(),
            "optimizer_actor": actor_optimizer.state_dict(),
            "optimizer_value_net": value_optimizer.state_dict()
        }
        os.makedirs(repositorio, exist_ok=True)
        torch.save(checkpoint, os.path.join(repositorio, "checkpoint.pth"))
        
        if iteration % 100 == 0:
            torch.save(world_model.state_dict(), os.path.join(repositorio, "world_model_weights.pth"))
            torch.save(actor.state_dict(), os.path.join(repositorio, "actor_weights.pth"))
            torch.save(value_net.state_dict(), os.path.join(repositorio, "value_net_weights.pth"))
            print("Model weights saved")
    
    print("\nTreinamento finalizado para essa fase!")
    
    torch.save(world_model.state_dict(), os.path.join(repositorio, "world_model_weights.pth"))
    torch.save(actor.state_dict(), os.path.join(repositorio, "actor_weights.pth"))
    torch.save(value_net.state_dict(), os.path.join(repositorio, "value_net_weights.pth"))
    print("Model weights saved")
    
    wandb.finish()

if __name__ == "__main__":
    main()
