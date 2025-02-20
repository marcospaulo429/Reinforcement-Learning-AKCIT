import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from wm_models import Autoencoder, TransitionModel, RewardModel
from dm_control import suite
from dm_control.suite.wrappers import pixels
from torch.utils.data import TensorDataset, DataLoader

def training_device():
    if torch.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else :
        device = torch.device("cpu")
        
    return device

def denormalize(img):
    return ((img + 1) * 127.5).clip(0, 255).astype(np.uint8)

def converter_cinza(pixels_arr):
    """
    Converte uma imagem RGB para escala de cinza.
    """
    img = Image.fromarray(pixels_arr)
    img_gray = img.convert("L")
    return np.array(img_gray)

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


def ver_reconstrucoes(world_model, test_loader, device, input_size, num_samples=10, 
                      action_dim=1, hidden_dim=256, HEIGHT=32, WIDTH=32):
    """
    Exibe lado a lado as próximas observações reais (target) e as reconstruções geradas pelo world_model.
    O test_loader deve retornar 4 tensores: (obs, action, reward, next_obs).

    Parâmetros:
      world_model: instância do DreamerWorldModel
      test_loader: DataLoader que retorna (obs, action, reward, next_obs)
      device: dispositivo (CPU/GPU) a ser usado
      input_size: dimensão da imagem flattenada (ex: HEIGHT*WIDTH)
      num_samples: número de imagens a exibir
      action_dim: dimensão da ação (necessária para consistência do forward)
      hidden_dim: dimensão do estado oculto da GRU
      HEIGHT, WIDTH: dimensões originais das imagens
    """
    obs, action, reward, next_obs = next(iter(test_loader))
    obs = obs.to(device)
    action = action.to(device)
    next_obs = next_obs.to(device)
    batch_size = obs.size(0)
    
    # Verifica e ajusta as dimensões de obs e next_obs
    if obs.dim() == 2:  # Formato: (B, HEIGHT*WIDTH)
        obs = obs.view(-1, 1, HEIGHT, WIDTH)
    elif obs.dim() == 3:  # Formato: (B, HEIGHT, WIDTH)
        obs = obs.unsqueeze(1)
        
    if next_obs.dim() == 2:
        next_obs = next_obs.view(-1, 1, HEIGHT, WIDTH)
    elif next_obs.dim() == 3:
        next_obs = next_obs.unsqueeze(1)
    
    # Inicializa o estado oculto com zeros
    prev_hidden = torch.zeros(batch_size, hidden_dim, device=device)
    
    with torch.no_grad():
        _, _, _, _, reward_pred, recon_next = world_model(obs, action, prev_hidden)
    
    original = next_obs.cpu().numpy()    
    reconstructed = recon_next.cpu().numpy()
    
    num_samples = min(num_samples, batch_size)
    plt.figure(figsize=(2 * num_samples, 4))
    
    for i in range(num_samples):
        plt.subplot(2, num_samples, i + 1)
        # Remover dimensões extras (ex.: canal) para visualização
        orig_img = original[i].squeeze()
        orig_img = denormalize(orig_img)
        plt.imshow(orig_img, cmap='gray')
        plt.axis('off')
        
        plt.subplot(2, num_samples, i + 1 + num_samples)
        recon_img = reconstructed[i].squeeze()
        recon_img = denormalize(recon_img)
        plt.imshow(recon_img, cmap='gray')
        plt.axis('off')
    
    plt.suptitle("Original (next_obs) vs. Reconstruída", fontsize=14)
    plt.tight_layout()
    plt.show()
