import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from dm_control import suite
from dm_control.suite.wrappers import pixels

from DreamerV1.utils.auxiliares import training_device, converter_cinza
from DreamerV1.utils.replay_buffer import ReplayBuffer
from DreamerV1.algorithms.general_algo import collect_replay_buffer, sample_data_sequences
from DreamerV1.algorithms.train_wm import DreamerWorldModel, get_data_loaders_from_replay_buffer, train_world_model
from DreamerV1.algorithms.behavior_learning import behavior_learning, extract_latent_sequences, create_latent_dataset
from actor_critic import ValueNet
from DreamerV1.models.models import ActionDecoder

import wandb


class DreamerTrainer:
    def __init__(self):
        # Parâmetros e Hiperparâmetros
        self.HEIGHT = 84
        self.WIDTH = 84
        self.hidden_dim = 256
        self.input_size = self.HEIGHT * self.WIDTH
        self.latent_dim = 256
        self.epochs_wm_behavior = 5
        self.update_step = 1
        self.batch_size = int(os.getenv('BATCH_SIZE', 1250))
        self.repository_path = os.getenv('MODEL_REPOSITORY_PATH', 'aluno_marcospaulo')
        self.num_iterations = int(os.getenv('NUM_ITERATIONS', 3))
        self.model_number = os.getenv('MODEL_NAME', 'dreamer_m1_run1')
        self.repositorio = f"{self.repository_path}/dreamer/"
        
        # Device e ambiente
        self.device = training_device()
        print(f"Usando device: {self.device}, repository_path: {self.repository_path}, "
              f"num_iterations: {self.num_iterations}, model_number: {self.model_number}, batch_size: {self.batch_size}")
        
        env = suite.load(domain_name="cartpole", task_name="swingup")
        self.env = pixels.Wrapper(env, pixels_only=True,
                                  render_kwargs={'height': self.HEIGHT, 'width': self.WIDTH, 'camera_id': 0})
        
        # Replay buffer e dimensão da ação
        self.replay_buffer = ReplayBuffer()
        self.action_dim = self.env.action_spec().shape[0]
        
        # Modelos e Otimizadores
        self.world_model = DreamerWorldModel(self.input_size, self.latent_dim, self.action_dim, self.hidden_dim).to(self.device)
        self.wm_optimizer = optim.Adam(self.world_model.parameters(), lr=6e-4)
        self.mse_loss = nn.MSELoss()
        
        self.actor = ActionDecoder(self.latent_dim, self.action_dim).to(self.device)
        self.value_net = ValueNet(self.latent_dim).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=8e-5)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=8e-5)
        
        # Carrega checkpoint, se existir
        self.checkpoint_path = os.path.join(self.repositorio, "checkpoint.pth")
        self.start_iteration = self.load_checkpoint(self.checkpoint_path)
        
        # Inicializa o wandb e define os hiperparâmetros
        wandb.init(project=self.model_number, config={
            "HEIGHT": self.HEIGHT,
            "WIDTH": self.WIDTH,
            "hidden_dim": self.hidden_dim,
            "latent_dim": self.latent_dim,
            "batch_size": self.batch_size,
            "epochs_wm_behavior": self.epochs_wm_behavior,
            "num_iterations": self.num_iterations,
            "update_step": self.update_step
        })
        wandb.define_metric("Reward/Episode", step_metric="global_step", summary="max")
        
        # Contador global para logs
        self.global_step = 0
        
        # Coleta episódios iniciais aleatórios para o replay_buffer
        self.replay_buffer = collect_replay_buffer(self.env, 5, self.replay_buffer)

    def load_checkpoint(self, checkpoint_path):
        if os.path.exists(checkpoint_path):
            print(f"Carregando checkpoint de {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)
            self.world_model.load_state_dict(checkpoint["world_model_state"])
            self.actor.load_state_dict(checkpoint["actor_state"])
            self.value_net.load_state_dict(checkpoint["value_net_state"])
            self.wm_optimizer.load_state_dict(checkpoint["optimizer_world_model"])
            self.actor_optimizer.load_state_dict(checkpoint["optimizer_actor"])
            self.value_optimizer.load_state_dict(checkpoint["optimizer_value_net"])
            iteration = checkpoint.get("iteration", 0)
            print(f"Checkpoint carregado. Retomando a partir da iteração {iteration}")
            return iteration
        else:
            print("Nenhum checkpoint encontrado. Iniciando treinamento do zero.")
            return 0

    def log_wandb(self, metrics):
        wandb.log(metrics, step=self.global_step)
        self.global_step += 1

    def log_weight_histograms(self, module, module_name):
        for name, param in module.named_parameters():
            wandb.log({f"{module_name}/{name}": wandb.Histogram(param.cpu().detach().numpy())},
                      step=self.global_step)

    def train(self):
        rewards_history = []
        for iteration in tqdm(range(self.start_iteration, self.num_iterations)):
            self.global_step += 1

            # Coleta e prepara dados do replay buffer
            data_sequence = sample_data_sequences(self.replay_buffer, num_sequences=50, sequence_length=50)
            train_loader, test_loader = get_data_loaders_from_replay_buffer(
                data_sequence, batch_size=self.batch_size, HEIGHT=self.HEIGHT, WIDTH=self.WIDTH)

            # Treinamento do World Model e Behavior
            for _ in range(self.update_step):
                # --- World Model ---
                if (iteration % 25 == 0) or (iteration < 25):
                    loss_train_history, loss_test_history, reward_train_history, reward_test_history = train_world_model(
                        10, self.world_model, train_loader, test_loader, self.device,
                        self.hidden_dim, self.mse_loss, self.wm_optimizer)
                    
                    # Log das métricas do world model
                    for epoch in range(self.epochs_wm_behavior):
                        self.log_wandb({
                            "WorldModel/TrainLoss": loss_train_history[epoch],
                            "WorldModel/TestLoss": loss_test_history[epoch],
                            "WorldModel/RewardTrainLoss": reward_train_history[epoch],
                            "WorldModel/RewardTestLoss": reward_test_history[epoch]
                        })
                
                # --- Behavior Learning ---
                epochs_behavior = self.epochs_wm_behavior
                latent_buffer = extract_latent_sequences(self.world_model, data_sequence, self.device)
                latent_dataset = create_latent_dataset(latent_buffer)
                latent_loader = DataLoader(latent_dataset, batch_size=self.batch_size)
                
                (self.actor, self.value_net, actor_loss_history, value_loss_history,
                 epoch_entropy_avg, epoch_mean_avg, epoch_std_avg) = behavior_learning(
                    self.world_model, self.actor, self.value_net,
                    latent_loader=latent_loader,
                    device=self.device,
                    horizon=15,
                    gamma=0.99,
                    value_optimizer=self.value_optimizer,
                    actor_optimizer=self.actor_optimizer,
                    mse_loss=self.mse_loss,
                    epochs_behavior=epochs_behavior
                )
                
                # Log das métricas do Behavior Learning
                for epoch in range(epochs_behavior):
                    self.log_wandb({
                        "Behavior/ActorLoss": actor_loss_history[epoch],
                        "Behavior/ValueLoss": value_loss_history[epoch],
                        "Behavior/EntropyAvg": epoch_entropy_avg[epoch],
                        "Behavior/MeanAvg": epoch_mean_avg[epoch],
                        "Behavior/StdAvg": epoch_std_avg[epoch],
                    })

            # --- Interação com o Ambiente ---
            time_step = self.env.reset()
            done = False
            obs_atual = converter_cinza(time_step.observation['pixels'])
            obs_atual = obs_atual.astype(np.float32) / 127.5 - 1.0

            episode_data = []
            steps_done = 0
            steps = 500
            ep_rewards = 0

            while steps_done < steps:
                obs_tensor = torch.tensor(obs_atual).view(1, 1, self.HEIGHT, self.WIDTH).to(self.device)
                conv_out = self.world_model.autoencoder.encoder_conv(obs_tensor)
                conv_out = conv_out.view(conv_out.size(0), -1)
                latent = self.world_model.autoencoder.encoder_fc(conv_out)

                dist, _, _ = self.actor(latent)
                action_tensor = dist.rsample() 
                action_np = action_tensor.detach().cpu().numpy()[0]

                time_step = self.env.step(action_np)
                done = time_step.last()
                reward = time_step.reward if time_step.reward is not None else 0.0
                ep_rewards += reward

                obs_prox = converter_cinza(time_step.observation['pixels'])
                obs_prox = obs_prox.astype(np.float32) / 127.5 - 1.0

                episode_data.append({
                    "obs": obs_atual,
                    "action": action_np,
                    "reward": reward,
                    "next_obs": obs_prox,
                    "done": done
                })
                obs_atual = obs_prox
                steps_done += 1

                if done:
                    self.replay_buffer.add_episode(episode_data)
                    episode_data = []
                    time_step = self.env.reset()
                    done = False
                    obs_atual = converter_cinza(time_step.observation['pixels'])
                    obs_atual = obs_atual.astype(np.float32) / 127.5 - 1.0

            rewards_history.append(ep_rewards)
            print(f"Recompensa acumulada = {ep_rewards:.2f}")
            self.log_wandb({"Reward/Episode": ep_rewards})
            
            # Log do histograma da distribuição do vetor latente (usando o primeiro batch do train_loader)
            for batch in test_loader:
                obs, _, _, _ = next(iter(train_loader))
                if obs.dim() == 2 and obs.size(1) == self.HEIGHT * self.WIDTH:
                    obs = obs.view(obs.size(0), 1, self.HEIGHT, self.WIDTH)
                obs = obs.to(self.device)
                conv_out = self.world_model.autoencoder.encoder_conv(obs)
                conv_out = conv_out.view(conv_out.size(0), -1)
                latent = self.world_model.autoencoder.encoder_fc(conv_out)
                self.log_wandb({"Latent/Distribution": wandb.Histogram(latent.cpu().detach().numpy())})
                break

        wandb.finish()


if __name__ == "__main__":
    trainer = DreamerTrainer()
    trainer.train()
