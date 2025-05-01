import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent)) 
import torch
import torch.nn as nn
import torch.nn.functional as f
import os
from utils.auxiliares import  training_device
device = training_device()


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
        x = f.relu(self.conv1(x))
        x = f.relu(self.conv2(x))
        x = f.relu(self.conv3(x))
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
        x = f.relu(self.deconv1(x))
        x = f.relu(self.deconv2(x))
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

    def load_checkpoint(self,checkpoint_dir,epoch_to_load, model, optimizer):
        """"Fazer resumo"""
        checkpoint_path = os.path.join(checkpoint_dir)

        # 4. Carregue o checkpoint
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
            
            # Carregue os estados
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Outras informações do checkpoint
            loaded_epoch = checkpoint['epoch']
            loaded_loss = checkpoint['loss']
            loaded_test_loss = checkpoint['test_loss']
            
            print(f"Checkpoint carregado com sucesso (Epoch {loaded_epoch})")
            print(f"Loss de treino no checkpoint: {loaded_loss:.4f}")
            print(f"Loss de teste no checkpoint: {loaded_test_loss:.4f}")
            
            # Coloque o modelo em modo de avaliação
            model.eval()

            return model
        else:
            print(f"Erro: Arquivo de checkpoint não encontrado em {checkpoint_path}")


def vae_loss_fn(encoder_inputs, vae_outputs, z_mean, z_log_var):
    
    reconstruction_loss = f.mse_loss(vae_outputs, encoder_inputs, reduction='sum') / encoder_inputs.size(0)
    
    # Cálculo da divergência KL
    kl_loss = -0.2 * torch.mean(1 + z_log_var - z_mean.pow(2) - torch.exp(z_log_var))
    
    return reconstruction_loss + kl_loss