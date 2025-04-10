import torch
import time
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from dm_control import suite
from replay_buffer import ReplayBuffer
from general_algo import collect_replay_buffer
from dm_control.suite.wrappers import pixels
from auxiliares import training_device
import torch
import numpy as np
from sklearn.model_selection import train_test_split
import itertools
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import wandb
import os


def reparameterize(z_mean, z_log_var):
    std = torch.exp(0.5 * z_log_var)
    eps = torch.randn_like(std)
    return z_mean + eps * std

# Encoder com mais camadas convolucionais para maior poder representacional
class CNNEncoder(nn.Module):
    def __init__(self, latent_dim, in_channels=1,hidden_units = 32):  
        super(CNNEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_units, kernel_size=4, stride=2, padding=1)   # -> (batch, 64, 42, 42)
        self.conv2 = nn.Conv2d(hidden_units, hidden_units*2, kernel_size=4, stride=2, padding=1)            # -> (batch, 128, 21, 21)
        self.conv3 = nn.Conv2d(hidden_units*2, hidden_units*4, kernel_size=4, stride=2, padding=1)           # -> (batch, 256, 10, 10)
        
        self.flatten_dim = hidden_units*4 * 10 * 10  # Tamanho do vetor achatado após as convoluções
        self.fc_mean = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_log_var = nn.Linear(self.flatten_dim, latent_dim)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Achata para (batch, flatten_dim)
        z_mean = self.fc_mean(x)
        z_log_var = self.fc_log_var(x)
        return z_mean, z_log_var

class CNNDecoder(nn.Module):
    def __init__(self, latent_dim, out_channels=1, hidden_units=32):  # Alterei o out_channels para 3 para imagens RGB
        super(CNNDecoder, self).__init__()
        self.hidden_units = hidden_units
        self.fc = nn.Linear(latent_dim, hidden_units*4 * 10 * 10)  # Mapeia o vetor latente para uma representação plana
        self.deconv1 = nn.ConvTranspose2d(hidden_units*4, hidden_units*2, kernel_size=4, stride=2, padding=1, output_padding=1)  
        self.deconv2 = nn.ConvTranspose2d(hidden_units*2, hidden_units, kernel_size=4, stride=2, padding=1, output_padding=0)
        self.deconv3 = nn.ConvTranspose2d(hidden_units, out_channels, kernel_size=4, stride=2, padding=1, output_padding=0)
    
    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), (self.hidden_units)*4, 10, 10)  # Reorganiza para (batch, 256, 10, 10)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = torch.tanh(self.deconv3(x))  # Sigmoid para valores entre 0 e 1
        return x


# VAE combinando o Encoder e o Decoder com CNNs mais complexas
class VAE(nn.Module):
    def __init__(self, latent_dim, in_channels=1, hidden_units=32):
        super(VAE, self).__init__()
        self.encoder = CNNEncoder(latent_dim, in_channels,hidden_units=hidden_units)
        self.decoder = CNNDecoder(latent_dim, out_channels=in_channels, hidden_units=hidden_units)
    
    def forward(self, x):
        z_mean, z_log_var = self.encoder(x)
        z = reparameterize(z_mean, z_log_var)
        recon = self.decoder(z)
        return recon, z_mean, z_log_var


def vae_loss_fn(encoder_inputs, vae_outputs, z_mean, z_log_var):
    
    # Cálculo do erro de reconstrução (MSE médio por amostra)
    reconstruction_loss = F.mse_loss(vae_outputs, encoder_inputs, reduction='sum') / encoder_inputs.size(0)
    
    # Cálculo da divergência KL
    # KL = -0.5 * mean(1 + log_var - z_mean^2 - exp(log_var))
    kl_loss = -0.2 * torch.mean(1 + z_log_var - z_mean.pow(2) - torch.exp(z_log_var))
    
    return reconstruction_loss + kl_loss

device = training_device()
data_len = 5 #TODO
batch_size = 5000#TODO

env = suite.load(domain_name="cartpole", task_name="swingup")
env = pixels.Wrapper(env, pixels_only=True,
                    render_kwargs={'height': 84, 'width': 84, 'camera_id': 0})

# Replay buffer e dimensão da ação
replay_buffer = ReplayBuffer()
# Coleta episódios iniciais aleatórios para o replay_buffer
replay_buffer = collect_replay_buffer(env, data_len, replay_buffer)

obs_list = []

for ep in replay_buffer.buffer:
    for step in ep:
        obs_list.append(step["obs"])

obs_array = np.array(obs_list)   # (N, 84, 84)

obs_tensor = torch.tensor(obs_array)

if obs_tensor.ndim != 4:
    obs_tensor = obs_tensor.unsqueeze(1)


train_obs, test_obs = train_test_split(obs_tensor, test_size=0.1, random_state=42)

# Criar datasets de treino e teste
train_dataset = TensorDataset(train_obs, train_obs)
test_dataset = TensorDataset(test_obs, test_obs)

# Criar DataLoader para treino e teste
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
def train_autoencoder_dreamer(num_epochs, latent_dim, in_channels, height, width, learning_rate, hidden_units, name_wandb, checkpoint_dir):
    # Inicializa o WandB com as configurações do experimento
    wandb.init(project="autoencoder_dreamer", name= name_wandb,config={
        "epochs": num_epochs,
        "latent_dim": latent_dim,
        "in_channels": in_channels,
        "height": height, 
        "width": width,
        "learning_rate": learning_rate,
        "hidden_units": hidden_units
    })

    model = VAE(latent_dim=latent_dim, in_channels=in_channels, hidden_units=hidden_units).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        # Loop de treinamento
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            recon, z_mean, z_log_var = model(inputs)
            loss = vae_loss_fn(inputs, recon, z_mean, z_log_var)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_loss = running_loss / (batch_idx + 1)
        
        # Avaliação no conjunto de teste
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs = inputs.to(device)
                targets = targets.to(device)
                recon, z_mean, z_log_var = model(inputs)
                loss = vae_loss_fn(inputs, recon, z_mean, z_log_var)
                test_loss += loss.item()
                
            avg_test_loss = test_loss / len(test_loader)  # Calcula a média da perda de teste
        
        # Seleciona imagens reais e suas reconstruções para plotar
        with torch.no_grad():
            inputs_batch, targets_batch = next(iter(test_loader))
            inputs_batch = inputs_batch.to(device)
            recon_batch, _, _ = model(inputs_batch)
            
            # Cria uma lista para armazenar as imagens logadas no WandB
            images = []
            for i in range(20):  # Seleciona 20 imagens do batch
                original_img = inputs_batch[i].cpu().squeeze()
                recon_img = recon_batch[i].cpu().squeeze()

                # Cria uma figura com duas subplots (imagem original e reconstruída)
                fig, axs = plt.subplots(1, 2, figsize=(8, 4))
                axs[0].imshow(original_img, cmap='gray')
                axs[0].set_title("Imagem Original")
                axs[0].axis("off")

                axs[1].imshow(recon_img, cmap='gray')
                axs[1].set_title("Reconstrução")
                axs[1].axis("off")

                # Adiciona a imagem ao log do WandB
                images.append(wandb.Image(fig))

                plt.close(fig)
        
        # Loga as 20 imagens e as métricas no WandB
        wandb.log({
            "epoch": epoch,
            "train_loss": avg_loss,
            "test_loss": avg_test_loss,
            "recon_images": images  # Loga as 20 imagens de teste
        })
            
        print(f"Fim da época {epoch+1}, Loss média de treino: {avg_loss}, Loss de teste: {avg_test_loss}")
        
        best_test_loss = 16
        if (epoch % 30 == 0) and (avg_loss < best_test_loss):  
            best_test_loss = avg_loss
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{name_wandb}_epoch{epoch}.pt")
            torch.save({#TODO
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'test_loss': avg_test_loss 
            }, checkpoint_path)
            
        del images, inputs_batch, targets_batch, original_img, recon_img, avg_loss, avg_test_loss, inputs, targets, recon, z_mean, z_log_var
        
    wandb.finish()
    
    del model, optimizer

        

# Lista de hiperparâmetros que você quer testar
latent_dim_list = [128] 
learning_rate_list = [1e-3, 1e-4] #TODO
hidden_units_list = [128]  

# Cria todas as combinações possíveis dos valores dos hiperparâmetros
param_combinations = list(itertools.product(latent_dim_list, learning_rate_list, hidden_units_list))

for latent_dim, learning_rate, hidden_units in param_combinations:
    print(f"Treinando com: latent_dim={latent_dim}, learning_rate={learning_rate}, hidden_units={hidden_units}")
    name_wandb = f"dgx_treino3-lat{latent_dim}_lr{learning_rate}_h{hidden_units}" #TODO
    
    if torch.cuda.is_available(): #TODO
        torch.cuda.empty_cache()
        
    # Chame a função de treinamento com os parâmetros da combinação
    train_autoencoder_dreamer(num_epochs=600, latent_dim=latent_dim, in_channels=1, 
                              height=84, width=84, learning_rate = learning_rate, hidden_units = hidden_units, name_wandb=name_wandb, #TODO
                              checkpoint_dir=f"/aluno_marcospaulo/{name_wandb}")#TODO
    time.sleep(15)
