# world_model.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from replay_buffer import ReplayBuffer, collect_replay_buffer
from wm_models import Autoencoder, TransitionModel, RewardModel
from dm_control import suite
from dm_control.suite.wrappers import pixels
from auxiliares import converter_cinza, get_data_loaders_from_replay_buffer, ver_reconstrucoes, denormalize, training_device
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid



class DreamerWorldModel(nn.Module):
    def __init__(self, input_size, latent_dim, action_dim, hidden_dim):
        """
        input_size: não é mais utilizado pelo autoencoder CNN, mas pode ser útil para compatibilidade.
        latent_dim: dimensão do vetor latente (por exemplo, 256)
        action_dim: dimensão da ação (por exemplo, 1)
        hidden_dim: dimensão do estado oculto da GRU (por exemplo, 256)
        """
        super(DreamerWorldModel, self).__init__()
        self.autoencoder = Autoencoder(latent_dim)
        self.transition_model = TransitionModel(latent_dim, action_dim, hidden_dim)
        self.reward_model = RewardModel(latent_dim)
        
    def forward(self, observation, action, prev_hidden):
        # observation deve ter shape (B, 1, 84, 84)
        # Usa o autoencoder CNN para obter o vetor latente a partir do encoder
        conv_out = self.autoencoder.encoder_conv(observation)  # (B, 64, 21, 21)
        conv_out = conv_out.view(conv_out.size(0), -1)           # (B, 64*21*21)
        latent = self.autoencoder.encoder_fc(conv_out)           # (B, latent_dim)
        
        latent_next, hidden, mean, std = self.transition_model(prev_hidden, latent, action)
        reward_pred = self.reward_model(latent_next)
        # Reconstrução usando o decoder do autoencoder:
        fc_out = self.autoencoder.decoder_fc(latent_next)        # (B, 64*21*21)
        fc_out = fc_out.view(-1, 64, 21, 21)                      # (B, 64, 21, 21)
        recon_obs = self.autoencoder.decoder_deconv(fc_out)       # (B, 1, 84, 84)
        return latent_next, hidden, mean, std, reward_pred, recon_obs


def train_autoencoder(autoencoder, train_loader, test_loader, device, num_epochs=10): 
    #writer = SummaryWriter()
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    for epoch in range(num_epochs):
        autoencoder.train()
        epoch_loss_train = 0.0
        for batch in train_loader:
            obs = batch[0].to(device)
            # (B, 1, 84, 84)
            if obs.dim() == 2:
                obs = obs.view(-1, 1, 84, 84)
            target = obs  
            output = autoencoder(obs)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss_train += loss.item()
        train_loss = epoch_loss_train / len(train_loader)
        #writer.add_scalar("Loss/train", loss, epoch)
        
        autoencoder.eval()
        epoch_loss_test = 0.0
        with torch.no_grad():
            for batch in test_loader:
                obs = batch[0].to(device)
                if obs.dim() == 2:
                    obs = obs.view(-1, 1, 84, 84)
                target = obs
                output = autoencoder(obs)
                loss = criterion(output, target)
                epoch_loss_test += loss.item()
        test_loss = epoch_loss_test / len(test_loader)
        #writer.add_scalar("Loss/test", loss, epoch)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")
    #writer.close()

        
def visualize_autoencoder(autoencoder, test_loader, device, HEIGHT, WIDTH, num_samples=8): 
    autoencoder.eval()
    batch = next(iter(test_loader))[0]
    batch = batch.to(device)
    
    # Se o batch estiver achatado (por exemplo, [B, 7056]) ou sem canal, redimensiona para [B, 1, HEIGHT, WIDTH]
    if batch.dim() == 2:  # formato (B, 7056)
        batch = batch.view(-1, 1, HEIGHT, WIDTH)
    elif batch.dim() == 3:  # formato (B, 84, 84) sem canal explícito
        batch = batch.unsqueeze(1)
    
    with torch.no_grad():
        output = autoencoder(batch)
    
    batch = batch.cpu().numpy()
    output = output.cpu().numpy()
    
    import matplotlib.pyplot as plt  # Caso ainda não esteja importado
    plt.figure(figsize=(2 * num_samples, 4))
    for i in range(num_samples):
        plt.subplot(2, num_samples, i + 1)
        # Se a imagem ainda estiver com canal, remova essa dimensão para visualização
        orig_img = batch[i].squeeze()  
        orig_img = denormalize(orig_img)
        plt.imshow(orig_img, cmap='gray')
        plt.axis('off')
        
        plt.subplot(2, num_samples, i + 1 + num_samples)
        rec_img = output[i].squeeze()
        rec_img = denormalize(rec_img)
        plt.imshow(rec_img, cmap='gray')
        plt.axis('off')
    
    plt.suptitle("Autoencoder - Original (acima) vs. Reconstruída (abaixo)")
    plt.tight_layout()
    plt.show()


def train_world_model(num_epochs, world_model, train_loader, test_loader, device, hidden_dim, mse_loss, wm_optimizer,repositorio): 
    writer = SummaryWriter(f"{repositorio}")
    for epoch in range(num_epochs):
        world_model.train()
        reward_loss_epoch = 0
        epoch_loss = 0.0
        train_batches = 0
        for batch in train_loader:
            obs, action, reward, next_obs = batch
            obs = obs.to(device)
            action = action.to(device)
            reward = reward.to(device)
            next_obs = next_obs.to(device)
            
            # Converte obs e next_obs para (B, 1, 84, 84) 
            if obs.dim() == 2:
                obs = obs.view(-1, 1, 84, 84)
            if next_obs.dim() == 2:
                next_obs = next_obs.view(-1, 1, 84, 84)
                
            batch_size_ = obs.size(0)
            prev_hidden = torch.zeros(batch_size_, hidden_dim, device=device)
            latent_next, hidden, mean, std, reward_pred, recon_next = world_model(obs, action, prev_hidden)
            
            recon_loss = mse_loss(recon_next, next_obs)
            reward_loss = mse_loss(reward_pred, reward)
            #kl_loss = torch.mean(-0.5 * torch.sum(1 + torch.log(std**2) - mean**2 - std**2, dim=-1))
            loss = recon_loss + reward_loss   #+ kl_loss 
            
            wm_optimizer.zero_grad()
            loss.backward()
            wm_optimizer.step()
            
            epoch_loss += loss.item()
            reward_loss_epoch += reward_loss
            train_batches += 1
        avg_train_loss = epoch_loss / train_batches
        writer.add_scalar("world_model_train/loss", avg_train_loss, epoch)
        writer.add_scalar("world_model_train/reward", reward_loss_epoch, epoch)
        
        world_model.eval()
        reward_loss_epoch = 0
        test_loss = 0.0
        test_batches = 0
        with torch.no_grad():
            for i,batch in enumerate(test_loader):
                obs, action, reward, next_obs = batch
                obs = obs.to(device)
                action = action.to(device)
                reward = reward.to(device)
                next_obs = next_obs.to(device)
                
                if obs.dim() == 2:
                    obs = obs.view(-1, 1, 84, 84)
                if next_obs.dim() == 2:
                    next_obs = next_obs.view(-1, 1, 84, 84)
                
                batch_size_ = obs.size(0)
                prev_hidden = torch.zeros(batch_size_, hidden_dim, device=device)
                latent_next, hidden, mean, std, reward_pred, recon_next = world_model(obs, action, prev_hidden)
                
                if i == 0:
                    real_imgs = next_obs[:10]
                    recon_imgs = recon_next[:10]
                    grid_real = make_grid(real_imgs, nrow=5, normalize=True, scale_each=True)
                    grid_recon = make_grid(recon_imgs, nrow=5, normalize=True, scale_each=True)
                    writer.add_image("Real Images", grid_real, epoch)
                    writer.add_image("Reconstructed Images", grid_recon, epoch)
            
                recon_loss = mse_loss(recon_next, next_obs)
                reward_loss = mse_loss(reward_pred, reward)
                #kl_loss = torch.mean(-0.5 * torch.sum(1 + torch.log(std**2) - mean**2 - std**2, dim=-1))
                loss = recon_loss + reward_loss  #+ kl_loss
                
                test_loss += loss.item()
                reward_loss_epoch += reward_loss
                test_batches += 1
        avg_test_loss = test_loss / test_batches
        writer.add_scalar("world_model_test/loss", avg_test_loss, epoch)
        writer.add_scalar("world_model_test/reward", reward_loss_epoch, epoch)
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f}")
    writer.close()

    

def main():
    HEIGHT = 84
    WIDTH = 84
    repositorio = "world_model/model_4"
    input_size = HEIGHT * WIDTH 
    latent_dim = 256
    action_dim = 1
    hidden_dim = 256
    num_epochs = 40
    batch_size = 32
    device = training_device()
    S = 14
    print("Using device:", device)
    
    env = suite.load(domain_name="cartpole", task_name="balance")
    env = pixels.Wrapper(env, pixels_only=True,
                         render_kwargs={'height': HEIGHT, 'width': WIDTH, 'camera_id': 0})
    
    # Coleta S episódios aleatórios
    replay_buffer = ReplayBuffer()
    
    replay_buffer = collect_replay_buffer(env, S, replay_buffer)
    
    train_loader, test_loader = get_data_loaders_from_replay_buffer(replay_buffer, batch_size=batch_size, test_split=0.1, HEIGHT=HEIGHT, WIDTH=WIDTH)
    
    #autoencoder = Autoencoder(latent_dim).to(device)
    #train_autoencoder(autoencoder, train_loader, test_loader, device, num_epochs=20)
    #visualize_autoencoder(autoencoder, test_loader, device, HEIGHT, WIDTH, num_samples=4)  # num_samples deve ser múltiplo do tamanho do batch
    
    world_model = DreamerWorldModel(input_size, latent_dim, action_dim, hidden_dim).to(device)
    wm_optimizer = optim.Adam(world_model.parameters(), lr=0.001)
    mse_loss = nn.MSELoss()
    
    train_world_model(num_epochs, world_model, train_loader, test_loader, device, hidden_dim, mse_loss, wm_optimizer,repositorio)
        
    torch.save(world_model.state_dict(), f"{repositorio}/world_model_weights.pth")
    print("Model weights saved to 'world_model_weights.pth'")
    
    with open(f"{repositorio}/training_details.txt", "a") as log_file:  
        log_file.write(f"Epochs: {num_epochs}, Batch: {batch_size}, Latent Dimension: {latent_dim}, Hidden Dimension: {hidden_dim}, Size Buffer: {S}\n")

    
    ver_reconstrucoes(world_model, test_loader, device, input_size, num_samples=8,
                      action_dim=action_dim, hidden_dim=hidden_dim, HEIGHT=HEIGHT, WIDTH=WIDTH)

if __name__ == "__main__":
    main()
