import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import yaml
from yaml import Loader


def training_device():
    if torch.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else :
        device = torch.device("cpu")
        
    print(device)
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

def load_config(config_path="config/config.yaml"):
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=Loader)
    return config