import torch
import torch.nn.functional as F
import random
import numpy as np # Importar se ainda não estiver importado

def test_world_model(
    global_step,
    writer,
    encoder,
    decoder,
    transition_model,
    reward_model,
    obs_seq,
    actions_seq,
    args,
    device,
    reparameterize
):
    print(f"--- Testando o World Model em global_step={global_step}: Previsão Multi-Passo ---")

    initial_real_obs = obs_seq[0, 0] / 255.0 # Primeira observação do primeiro ambiente
    
    with torch.no_grad():
        mu_init, logvar_init = encoder(initial_real_obs.unsqueeze(0))
        initial_latent_sample = reparameterize(mu_init, logvar_init) 

    imagined_state = {
        'sample': initial_latent_sample,
        'rnn_state': torch.zeros(1, transition_model.belief_size, device=device),
        'mean': initial_latent_sample,
        'stddev': torch.ones_like(initial_latent_sample),
        'belief': torch.zeros(1, transition_model.belief_size, device=device)
    }

    imagined_images = []
    imagined_rewards_pred = []
    
    # Adicionar a imagem inicial real (reconstruída) à lista
    with torch.no_grad():
        initial_reconstruction = decoder(initial_latent_sample)
        
        # Ajuste a imagem reconstruída para o formato do TensorBoard (3 canais RGB)
        reconstructed_initial_rgb = initial_reconstruction.squeeze(0)[-1:].repeat(3, 1, 1).cpu()
        if reconstructed_initial_rgb.shape[0] != 3:
            reconstructed_initial_rgb = reconstructed_initial_rgb.mean(dim=0, keepdim=True).repeat(3, 1, 1)
        
        imagined_images.append(reconstructed_initial_rgb)
        imagined_rewards_pred.append(torch.tensor([0.0])) # Recompensa 0 para o passo inicial

    # Loop de imaginação
    for k in range(args.horizon_to_imagine):
        with torch.no_grad():
            # Escolha uma ação aleatória para o passo imaginado
            imagined_action_id = torch.tensor([random.randint(0, args.action_dim - 1)], device=device)
            
            # Converter a ação escolhida para one-hot encoding
            imagined_action_one_hot = F.one_hot(imagined_action_id, num_classes=args.action_dim).float()

            # Avance o World Model no espaço latente usando a ação imaginada
            imagined_state = transition_model._transition(imagined_state, imagined_action_one_hot)

            # Preveja a recompensa para o estado imaginado
            predicted_reward_step = reward_model(imagined_state['belief'], imagined_state['sample']).cpu().squeeze(-1)
            
            # Reconstrua a imagem do estado imaginado usando o Decoder
            reconstructed_image_step = decoder(imagined_state['sample'])
            
            # Ajuste a imagem reconstruída para o formato TensorBoard (3 canais RGB)
            reconstructed_image_rgb = reconstructed_image_step.squeeze(0)[-1:].repeat(3, 1, 1).cpu()
            if reconstructed_image_rgb.shape[0] != 3:
                reconstructed_image_rgb = reconstructed_image_rgb.mean(dim=0, keepdim=True).repeat(3, 1, 1)
            
            imagined_images.append(reconstructed_image_rgb)
            imagined_rewards_pred.append(predicted_reward_step)

    # --- Substituição do vídeo pela concatenação de imagens ---
    # Empilhar todas as imagens imaginadas horizontalmente
    # imagined_images é uma lista de tensores [C, H, W]
    # Queremos um tensor final [C, H, W_total] onde W_total = Horizon * W
    
    if imagined_images: # Garante que a lista não está vazia
        # Concatena as imagens ao longo da largura (dim=2)
        concatenated_images_tensor = torch.cat(imagined_images, dim=2)
    else:
        print("Aviso: Nenhuma imagem imaginada para concatenar.")

    # --- Logging de recompensas (mantido) ---
    imagined_rewards_pred_tensor = torch.cat(imagined_rewards_pred)

    print(f"--- Teste do World Model concluído para global_step={global_step} ---")
    return concatenated_images_tensor, imagined_rewards_pred_tensor.sum().item()