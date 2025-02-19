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
from auxiliares import converter_cinza, get_data_loaders_from_replay_buffer, ver_reconstrucoes, denormalize

class DreamerWorldModel(nn.Module):
    def __init__(self, input_size, latent_dim, action_dim, hidden_dim):
        super(DreamerWorldModel, self).__init__()
        self.autoencoder = Autoencoder(input_size,latent_dim)
        self.transition_model = TransitionModel(latent_dim, action_dim, hidden_dim)
        self.reward_model = RewardModel(latent_dim)
        
    def forward(self, observation, action, prev_hidden):
        latent = self.autoencoder.encoder(observation)
        latent_next, hidden, mean, std = self.transition_model(prev_hidden, latent, action)
        reward_pred = self.reward_model(latent_next)
        recon_obs = self.autoencoder.decoder(latent_next)
        return latent_next, hidden, mean, std, reward_pred, recon_obs


def train_autoencoder(autoencoder, train_loader, test_loader, device, num_epochs=10): 
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    for epoch in range(num_epochs):
        autoencoder.train()
        epoch_loss_train = 0.0
        for batch in train_loader:
            obs = batch[0].to(device)
            target = obs  
            output = autoencoder(obs)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss_train += loss.item()
        train_loss = epoch_loss_train / len(train_loader)
        
        autoencoder.eval()
        epoch_loss_test = 0.0
        with torch.no_grad():
            for batch in test_loader:
                obs = batch[0].to(device)
                target = obs
                output = autoencoder(obs)
                loss = criterion(output, target)
                epoch_loss_test += loss.item()
        test_loss = epoch_loss_test / len(test_loader)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")

        
def visualize_autoencoder(autoencoder, test_loader, device, HEIGHT, WIDTH, num_samples=8): 
    autoencoder.eval()
    batch = next(iter(test_loader))[0]
    batch = batch.to(device)
    with torch.no_grad():
        output = autoencoder(batch)
    batch = batch.cpu().numpy()
    output = output.cpu().numpy()
    
    plt.figure(figsize=(2 * num_samples, 4))
    for i in range(num_samples):
        plt.subplot(2, num_samples, i + 1)
        orig_img = batch[i].reshape(HEIGHT, WIDTH)
        orig_img = denormalize(orig_img)
        plt.imshow(orig_img, cmap='gray')
        plt.axis('off')
        
        plt.subplot(2, num_samples, i + 1 + num_samples)
        rec_img = output[i].reshape(HEIGHT, WIDTH)
        rec_img = denormalize(rec_img)
        plt.imshow(rec_img, cmap='gray')
        plt.axis('off')
    
    plt.suptitle("Autoencoder - Original (acima) vs. Reconstruída (abaixo)")
    plt.tight_layout()
    plt.show()

def train_world_model(num_epochs, world_model, train_loader, test_loader, device, hidden_dim, mse_loss, wm_optimizer): 
    for epoch in range(num_epochs):
        world_model.train()
        epoch_loss = 0.0
        train_batches = 0
        for batch in train_loader:
            obs, action, reward, next_obs = batch
            obs = obs.to(device)
            action = action.to(device)
            reward = reward.to(device)
            next_obs = next_obs.to(device)
            
            batch_size_ = obs.size(0)
            prev_hidden = torch.zeros(batch_size_, hidden_dim, device=device)
            
            latent_next, hidden, mean, std, reward_pred, recon_next = world_model(obs, action, prev_hidden)
            recon_loss = mse_loss(recon_next, next_obs)
            reward_loss = mse_loss(reward_pred, reward)
            #kl_loss = torch.mean(-0.5 * torch.sum(1 + torch.log(std**2) - mean**2 - std**2, dim=-1))
            loss = recon_loss + reward_loss #+ kl_loss
            
            wm_optimizer.zero_grad()
            loss.backward()
            wm_optimizer.step()
            
            epoch_loss += loss.item()
            train_batches += 1
        avg_train_loss = epoch_loss / train_batches
        
        world_model.eval()
        test_loss = 0.0
        test_batches = 0
        with torch.no_grad():
            for batch in test_loader:
                obs, action, reward, next_obs = batch
                obs = obs.to(device)
                action = action.to(device)
                reward = reward.to(device)
                next_obs = next_obs.to(device)
                
                batch_size_ = obs.size(0)
                prev_hidden = torch.zeros(batch_size_, hidden_dim, device=device)
                latent_next, hidden, mean, std, reward_pred, recon_next = world_model(obs, action, prev_hidden)
                recon_loss = mse_loss(recon_next, next_obs)
                reward_loss = mse_loss(reward_pred, reward)
                #kl_loss = torch.mean(-0.5 * torch.sum(1 + torch.log(std**2) - mean**2 - std**2, dim=-1))
                loss = recon_loss + reward_loss #+ kl_loss
                
                test_loss += loss.item()
                test_batches += 1
        avg_test_loss = test_loss / test_batches
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f}")
    

def main():
    HEIGHT = 84
    WIDTH = 84
    input_size = HEIGHT * WIDTH
    latent_dim = 512
    action_dim = 1
    hidden_dim = 512
    num_epochs = 10
    batch_size = 4
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    print("Using device:", device)
    
    env = suite.load(domain_name="cartpole", task_name="balance")
    env = pixels.Wrapper(env, pixels_only=True,
                         render_kwargs={'height': HEIGHT, 'width': WIDTH, 'camera_id': 0})
    
    # Coleta S episódios aleatórios
    replay_buffer = ReplayBuffer()
    
    S = 5
    replay_buffer = collect_replay_buffer(env,S,replay_buffer)
    
    #replay_buffer.save_to_csv()
    #replay_buffer.load_from_csv()
    
    train_loader, test_loader = get_data_loaders_from_replay_buffer(replay_buffer, batch_size=batch_size, test_split=0.1, HEIGHT=HEIGHT, WIDTH=WIDTH)
    
    autoencoder = Autoencoder(input_size=HEIGHT*WIDTH, latent_dim=latent_dim).to(device)
    
    train_autoencoder(autoencoder, train_loader, test_loader, device, num_epochs=20)
    
    visualize_autoencoder(autoencoder, test_loader, device, HEIGHT, WIDTH, num_samples=4)#num_samples deve ser multiplo do tamanho do batch
    
    # Instancia o World Model
    world_model = DreamerWorldModel(input_size, latent_dim, action_dim, hidden_dim).to(device)
    wm_optimizer = optim.Adam(world_model.parameters(), lr=0.001)
    mse_loss = nn.MSELoss()
    
    #world_model = train_world_model(num_epochs, world_model, train_loader, test_loader, device, hidden_dim, mse_loss, wm_optimizer)
    
    #torch.save(world_model.state_dict(), "world_model_weights.pth")
    #print("Model weights saved to 'world_model_weights.pth'")
    
    #ver_reconstrucoes(world_model, test_loader, device, input_size, num_samples=8,
         #             action_dim=action_dim, hidden_dim=hidden_dim, HEIGHT=HEIGHT, WIDTH=WIDTH)

if __name__ == "__main__":
    main()
