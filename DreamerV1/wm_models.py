import torch
import torch.nn as nn

# --- Autoencoder baseado em CNN ---
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