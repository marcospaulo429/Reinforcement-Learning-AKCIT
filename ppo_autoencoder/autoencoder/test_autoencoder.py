import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import torch.nn.init as init

# Inicializar com valores aleatórios a partir de uma distribuição normal
def init_weights(layer):
    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
        init.normal_(layer.weight, mean=0.0, std=0.1)  # Média 0 e desvio padrão 0.1
        if layer.bias is not None:
            init.zeros_(layer.bias)  # Inicializar o bias com 0



# ---------- layer_init ----------
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

# ---------- VAE ----------
class VAE(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder_cnn = nn.Sequential(
            layer_init(nn.Conv2d(4, 16, kernel_size=8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(16, 32, kernel_size=4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 32, kernel_size=3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.encoder_fc_mu = layer_init(nn.Linear(32 * 7 * 7, latent_dim))
        self.encoder_fc_logvar = layer_init(nn.Linear(32 * 7 * 7, latent_dim))

        self.decoder_fc = layer_init(nn.Linear(latent_dim, 32 * 7 * 7))
        self.decoder_deconv = nn.Sequential(
            nn.ReLU(),
            nn.Unflatten(1, (32, 7, 7)),
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 4, kernel_size=8, stride=4),
            nn.Sigmoid(),
        )

        self.apply(init_weights)  # Inicializar pesos uniformemente

    def encode(self, x):
        x = self.encoder_cnn(x)
        mu = self.encoder_fc_mu(x)
        logvar = self.encoder_fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.decoder_fc(z)
        return self.decoder_deconv(x)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar, z

# ---------- Função de perda ----------
def vae_loss(recon_x, x, mu, logvar):
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl

# ---------- TensorBoard ----------
writer = SummaryWriter(log_dir="runs/vae_fashionmnist")

# ---------- Dataset e DataLoaders ----------
transform = transforms.Compose([
    transforms.Resize((84, 84)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(4, 1, 1))  # MNIST tem 1 canal → repetir para 4
])

train_data = datasets.FashionMNIST(root="./data/fashion", train=True, download=True, transform=transform)
test_data = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)
#train_data = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
#test_data = datasets.MNIST(root="./data/fashion", train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_loader = DataLoader(test_data, batch_size=128, shuffle=False)

# ---------- Treinamento ----------
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
model = VAE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Checkpoint path
checkpoint_path = "vae_checkpoint2.pth"

# Carregar modelo e otimizador de um checkpoint, se disponível
start_epoch = 1
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1  # Iniciar da próxima época
    print(f"Resuming from epoch {start_epoch}...")

for epoch in range(start_epoch, 100):
    model.train()
    train_loss = 0
    for batch, _ in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        recon, mu, logvar, _ = model(batch)
        loss = vae_loss(recon, batch, mu, logvar)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader.dataset)
    writer.add_scalar("Loss/Train", avg_train_loss, epoch)

    # ---------- Validação ----------
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch, _ in test_loader:
            batch = batch.to(device)
            recon, mu, logvar, _ = model(batch)
            loss = vae_loss(recon, batch, mu, logvar)
            test_loss += loss.item()

    avg_test_loss = test_loss / len(test_loader.dataset)
    writer.add_scalar("Loss/Test", avg_test_loss, epoch)

    print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f}")

    # ---------- Reconstrução visual no TensorBoard ----------
    with torch.no_grad():
        test_batch, _ = next(iter(test_loader))
        test_batch = test_batch[:10].to(device)
        recon, _, _, _ = model(test_batch)
        # Apenas 1 canal para visualização
        originals = test_batch[:, 0:1]
        reconstructions = recon[:, 0:1]
        comparison = torch.cat([originals.cpu(), reconstructions.cpu()])
        grid = utils.make_grid(comparison, nrow=10, normalize=True, pad_value=1)
        writer.add_image("Reconstruction", grid, epoch)

    # Salvar checkpoint após cada época
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)

writer.close()
