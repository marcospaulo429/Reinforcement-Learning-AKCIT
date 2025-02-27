# world_model.py

import torch
import torch.nn as nn
import torch.optim as optim
from replay_buffer import ReplayBuffer, collect_replay_buffer
from wm_models import DreamerWorldModel
from dm_control import suite
from dm_control.suite.wrappers import pixels
from auxiliares import  get_data_loaders_from_replay_buffer, denormalize, training_device
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid


def train_autoencoder(autoencoder, train_loader, test_loader, device, num_epochs=10): 
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    train_loss_history = []
    test_loss_history = []
    
    for epoch in range(num_epochs):
        autoencoder.train()
        epoch_loss_train = 0.0
        for batch in train_loader:
            obs = batch[0].to(device)
            # Garante que a imagem tenha shape (B, 1, 84, 84)
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
        train_loss_history.append(train_loss)
        
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
        test_loss_history.append(test_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")
        
    return train_loss_history, test_loss_history
        
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


def main():
    HEIGHT = 84
    WIDTH = 84
    repositorio = "world_model/model_7"
    input_size = HEIGHT * WIDTH 
    latent_dim = 256
    action_dim = 1
    hidden_dim = 256
    num_epochs = 10
    batch_size = 32
    device = training_device()
    S = 8
    print("Using device:", device)
    
    writer = SummaryWriter(log_dir=repositorio)
    
    env = suite.load(domain_name="cartpole", task_name="balance")
    env = pixels.Wrapper(env, pixels_only=True,
                         render_kwargs={'height': HEIGHT, 'width': WIDTH, 'camera_id': 0})
    
    replay_buffer = ReplayBuffer()
    replay_buffer = collect_replay_buffer(env, S, replay_buffer)
    
    train_loader, test_loader = get_data_loaders_from_replay_buffer(
        replay_buffer, batch_size=batch_size, test_split=0.1, HEIGHT=HEIGHT, WIDTH=WIDTH)
    
    """# ----- Treinamento do Autoencoder -----
    autoencoder = Autoencoder(latent_dim).to(device)
    # Treina o autoencoder por 20 épocas
    train_loss_history, test_loss_history = train_autoencoder(autoencoder, train_loader, test_loader, device, num_epochs)
    
    # Registra as perdas do autoencoder no TensorBoard
    for epoch, (train_loss, test_loss) in enumerate(zip(train_loss_history, test_loss_history)):
        writer.add_scalar("Autoencoder/TrainLoss", train_loss, epoch)
        writer.add_scalar("Autoencoder/TestLoss", test_loss, epoch) """
    
    # Visualiza as reconstruções do autoencoder
    #visualize_autoencoder(autoencoder, test_loader, device, HEIGHT, WIDTH, num_samples=4)
    
    # ----- Treinamento do World Model -----
    world_model = DreamerWorldModel(input_size, latent_dim, action_dim, hidden_dim).to(device)
    wm_optimizer = optim.Adam(world_model.parameters(), lr=0.001)
    mse_loss = nn.MSELoss()
    
    loss_train_history, loss_test_history, reward_train_history, reward_test_history = train_world_model(
        num_epochs, world_model, train_loader, test_loader, device, hidden_dim, mse_loss, wm_optimizer, writer)
    
    for epoch in range(num_epochs):
        writer.add_scalar("WorldModel/TrainLoss", loss_train_history[epoch], epoch)
        writer.add_scalar("WorldModel/TestLoss", loss_test_history[epoch], epoch)
        writer.add_scalar("WorldModel/RewardTrainLoss", reward_train_history[epoch], epoch)
        writer.add_scalar("WorldModel/RewardTestLoss", reward_test_history[epoch], epoch)
    
    torch.save(world_model.state_dict(), f"{repositorio}/world_model_weights.pth")
    print("Model weights saved to 'world_model_weights.pth'")
    
    with open(f"{repositorio}/training_details.txt", "a") as log_file:
        log_file.write(f"Epochs: {num_epochs}, Batch: {batch_size}, Latent Dimension: {latent_dim}, Hidden Dimension: {hidden_dim}, Size Buffer: {S}\n")
    
    writer.close()

if __name__ == "__main__":
    main()
