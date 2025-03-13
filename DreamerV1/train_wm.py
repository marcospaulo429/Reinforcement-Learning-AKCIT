import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, latent_dim):
        """
        Constrói um autoencoder com arquitetura CNN.
        Entrada: imagem em escala de cinza com shape (B, 1, 84, 84)
        Saída: imagem reconstruída com shape (B, 1, 84, 84)
        """
        super(Autoencoder, self).__init__()
        # Encoder: convoluções para extrair features
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # (B, 32, 42, 42)
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # (B, 64, 21, 21)
            nn.ReLU(True)
        )
        # Camada fully-connected para gerar o vetor latente
        self.encoder_fc = nn.Linear(64 * 21 * 21, latent_dim)
        
        # Decoder: transforma o vetor latente de volta em feature maps
        self.decoder_fc = nn.Linear(latent_dim, 64 * 21 * 21)
        self.decoder_deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # (B, 32, 42, 42)
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),   # (B, 1, 84, 84)
            nn.Tanh()  # saída em [-1,1]
        )
        
    def forward(self, x):
        # x deve ter shape (B, 1, 84, 84)
        conv_out = self.encoder_conv(x)              # (B, 64, 21, 21)
        conv_out = conv_out.view(conv_out.size(0), -1) # (B, 64*21*21)
        latent = self.encoder_fc(conv_out)             # (B, latent_dim)
        fc_out = self.decoder_fc(latent)               # (B, 64*21*21)
        fc_out = fc_out.view(-1, 64, 21, 21)            # (B, 64, 21, 21)
        reconstructed = self.decoder_deconv(fc_out)    # (B, 1, 84, 84)
        return reconstructed

class TransitionModel(nn.Module):
    def __init__(self, latent_dim, action_dim, hidden_dim):
        super(TransitionModel, self).__init__()
        self.gru = nn.GRU(latent_dim + action_dim, hidden_dim, batch_first=True)
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_std  = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, prev_hidden, latent, action):
        x = torch.cat([latent, action], dim=-1).unsqueeze(1)  # [batch, 1, latent_dim+action_dim]
        output, hidden = self.gru(x, prev_hidden.unsqueeze(0))
        hidden = hidden.squeeze(0)  # [batch, hidden_dim]
        mean = self.fc_mean(hidden)
        std  = torch.exp(self.fc_std(hidden))
        eps = torch.randn_like(std)
        latent_next = mean + eps * std
        return latent_next, hidden, mean, std

class RewardModel(nn.Module):
    def __init__(self, latent_dim):
        super(RewardModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    def forward(self, latent):
        return self.fc(latent)
    
    
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

def get_data_loaders_from_replay_buffer(replay_buffer, batch_size=64, test_split=0.1, HEIGHT=84, WIDTH=84, action_dim=1):
    """
    Extrai dados do replay_buffer para treinamento supervisionado do World Model.
    Cada exemplo contém: obs, action, reward, next_obs.
    As imagens (obs e next_obs) estão em formato (HEIGHT, WIDTH) e são flattenadas para (HEIGHT*WIDTH).
    Garante que o array de ação tenha shape (N, action_dim).
    """
    obs_list = []
    action_list = []
    reward_list = []
    next_obs_list = []
    
    for ep in replay_buffer.buffer:
        for step in ep:
            obs_list.append(step["obs"])
            action_list.append(step["action"])
            reward_list.append(step["reward"])
            next_obs_list.append(step["next_obs"])
    
    # Converte para arrays NumPy
    obs_array = np.array(obs_list)           # (N, HEIGHT, WIDTH)
    next_obs_array = np.array(next_obs_list)   # (N, HEIGHT, WIDTH)
    obs_array = obs_array.astype(np.float32)
    next_obs_array = next_obs_array.astype(np.float32)
    
    # (N, HEIGHT, WIDTH) -> (N, HEIGHT*WIDTH)
    N, H, W = obs_array.shape
    obs_array = obs_array.reshape(N, H * W)
    next_obs_array = next_obs_array.reshape(N, H * W)
    
    action_array = np.array(action_list).astype(np.float32)
    # array passa a ter (N, action_dim)
    action_array = action_array.reshape(-1, action_dim)
    
    reward_array = np.array(reward_list).astype(np.float32).reshape(-1, 1)
    
    dataset = TensorDataset(torch.tensor(obs_array),
                            torch.tensor(action_array),
                            torch.tensor(reward_array),
                            torch.tensor(next_obs_array))
    
    indices = np.arange(N)
    np.random.shuffle(indices)
    split = int((1 - test_split) * N)
    train_indices = indices[:split]
    test_indices = indices[split:]
    
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def train_world_model(num_epochs, world_model, train_loader, test_loader, device, hidden_dim, mse_loss, wm_optimizer): 

    reward_train_history = []
    reward_test_history = []
    loss_train_history = []
    loss_test_history = []
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
            
            if obs.dim() == 2:
                obs = obs.view(-1, 1, 84, 84)
            if next_obs.dim() == 2:
                next_obs = next_obs.view(-1, 1, 84, 84)
                
            batch_size_ = obs.size(0)
            prev_hidden = torch.zeros(batch_size_, hidden_dim, device=device)
            latent_next, hidden, mean, std, reward_pred, recon_next = world_model(obs, action, prev_hidden)
            
            recon_loss = mse_loss(recon_next, next_obs)
            reward_loss = mse_loss(reward_pred, reward)
            # kl_loss = torch.mean(-0.5 * torch.sum(1 + torch.log(std**2) - mean**2 - std**2, dim=-1))
            loss = recon_loss + reward_loss  # + kl_loss 
            
            wm_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(world_model.parameters(), max_norm=100)
            wm_optimizer.step()
            
            epoch_loss += loss.item()
            reward_loss_epoch += reward_loss.item()  # converter para float
            train_batches += 1
        avg_train_loss = epoch_loss / train_batches
        loss_train_history.append(avg_train_loss)
        reward_train_history.append(reward_loss_epoch)
        
        world_model.eval()
        reward_loss_epoch = 0
        test_loss = 0.0
        test_batches = 0
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
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
                
                """if i == 0: #TODO
                    # Seleciona as 10 primeiras imagens reais e reconstruídas
                    real_imgs = next_obs[:10]
                    recon_imgs = recon_next[:10]
                    # Cria uma grid (opcionalmente você pode usar make_grid do torchvision)
                    grid_real = make_grid(real_imgs, nrow=5, normalize=True, scale_each=True)
                    grid_recon = make_grid(recon_imgs, nrow=5, normalize=True, scale_each=True)
                    # Loga as imagens no wandb usando wandb.Image
                    wandb.log({
                        "Real Images": wandb.Image(grid_real),
                        "Reconstructed Images": wandb.Image(grid_recon)
                    }) """
            
                recon_loss = mse_loss(recon_next, next_obs)
                reward_loss = mse_loss(reward_pred, reward)
                # kl_loss = torch.mean(-0.5 * torch.sum(1 + torch.log(std**2) - mean**2 - std**2, dim=-1))
                loss = recon_loss + reward_loss  # + kl_loss
                
                test_loss += loss.item()
                reward_loss_epoch += reward_loss.item()
                test_batches += 1
        avg_test_loss = test_loss / test_batches
        loss_test_history.append(avg_test_loss)
        reward_test_history.append(reward_loss_epoch)
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f}")
    
    return loss_train_history, loss_test_history, reward_train_history, reward_test_history

