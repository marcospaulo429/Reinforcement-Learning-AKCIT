import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from dm_control import suite
from replay_buffer import ReplayBuffer
from general_algo import collect_replay_buffer
from dm_control.suite.wrappers import pixels
from train_wm import DreamerWorldModel
from actor_critic import ValueNet
from models import ActionDecoder
import wandb
from auxiliares import training_device

torch.autograd.set_detect_anomaly(True)

def compute_v(rewards, values, tau, k, gamma=0.99, lamb=0.95, t=0):
    """
    v_lambda : (1,B)
    """
    horizon, B = rewards.shape
    h = min(tau + k, t + horizon - 1)
    
    v = torch.zeros(1, B, dtype=rewards.dtype, device=rewards.device)
    
    for n in range(tau, h - 1):
        v += (gamma ** (n - tau)) * rewards[n] + (gamma ** (h - tau) * values[h])
    return v

def compute_v_lambda(rewards, values, tau, gamma=0.99, lamb=0.95):
    horizon, B = rewards.shape
    v_lambda = torch.zeros(1, B, dtype=rewards.dtype, device=rewards.device)
    
    for n in range(1, horizon - 1):
        v_lambda += (lamb ** (n - 1) * compute_v(rewards, values, tau, n)) + (lamb ** (horizon - 1)) * compute_v(rewards, values, tau, horizon)
        
    v_lambda = (1 - lamb) * v_lambda
    return v_lambda

def behavior_learning(
    data_sequence, world_model, batch, horizon,
    actor, value_net, device, lambda_=0.95,
    value_optimizer=None, actor_optimizer=None
):
    """
    Exemplo de função que:
      1) Carrega dados reais (obs, actions).
      2) Faz rollouts imaginados (com world_model) de tamanho 'horizon'.
      3) Calcula V_lambda para cada item do batch.
      4) Atualiza rede de valor e ator (caso otimizadores sejam fornecidos).
    """
    obs_list = []
    action_list = []
    
    for ep in data_sequence.buffer:
        for step in ep:
            obs_list.append(step["obs"])
            action_list.append(step["action"])
    
    obs_array = np.array(obs_list)   # (N, 84, 84)
    action_array = np.array(action_list)  # (N,)
    
    # Adiciona canal de cor
    obs_array = np.expand_dims(obs_array, axis=1)  # (N, 1, 84, 84)
    
    # Cria tensores diretamente no device selecionado
    obs_tensor = torch.tensor(obs_array, dtype=torch.float32, device=device)
    action_tensor = torch.tensor(action_array, dtype=torch.float32, device=device)
    
    dataset = TensorDataset(obs_tensor, action_tensor)
    data_loader = DataLoader(dataset, batch_size=batch, shuffle=True)
        
    for batch_obs, batch_action in data_loader:
        # Caso os dados ainda não estejam no device
        batch_obs = batch_obs.to(device)
        batch_action = batch_action.to(device)
        
        batch_size = batch_obs.size(0)
        hidden_dim = world_model.transition_model.gru.hidden_size
        
        # Inicializa o estado oculto no device
        prev_hidden = torch.zeros(batch_size, hidden_dim, device=device)
        
        # Passo 0: gerar latente e recompensa do "passo real"
        latent, hidden, mean, std, reward, recon_obs = world_model(batch_obs, batch_action, prev_hidden)
        
        # Armazenar recompensas e valores para depois calcular V_lambda
        rewards = [(reward.detach()).view(-1)]  
        values = [value_net(latent.detach()).view(-1)]  
                
        # Rollout imaginado
        current_obs = recon_obs  # O world_model gera uma "próxima observação imaginada"
        for i in range(horizon):
            # Ação do ator (pode ser determinístico ou amostrado)
            current_action = actor(latent).rsample()
            
            # Próximo passo imaginado
            latent, hidden, mean, std, reward, recon_obs = world_model(current_obs, current_action, hidden)
        
            # Armazena reward e value
            rewards.append((reward.detach()).view(-1))
            values.append(value_net(latent.detach()).view(-1))
            
            # Atualiza a observação
            current_obs = recon_obs
                    
        # Empilha tudo => (T, B), onde T = horizon + 1
        rewards_tensor = torch.stack(rewards, dim=0)  # shape (T, B)
        values_tensor  = torch.stack(values, dim=0)    # shape (T, B)

        v_lambda_actor = torch.zeros(1, batch_size, dtype=rewards_tensor.dtype, device=rewards_tensor.device, requires_grad=True)
        for tau in range(horizon):
            v_lam_tau = compute_v_lambda(rewards_tensor[tau:], values_tensor[tau:], tau=0, gamma=0.99, lamb=0.95)
            v_lambda_actor = v_lambda_actor + v_lam_tau
            
        actor_loss = -v_lambda_actor.mean()
        actor_loss.backward(retain_graph=True)
        actor_optimizer.step()
                
        v_lambda_value = torch.zeros(1, batch_size, dtype=rewards_tensor.dtype, device=rewards_tensor.device, requires_grad=True)
        for tau in range(horizon):
            v_lam_tau = compute_v_lambda(rewards_tensor[tau:], values_tensor[tau:], tau=0, gamma=0.99, lamb=0.95)
            v_lambda_value = (((values_tensor[tau] - v_lam_tau) ** 2) / 2) + v_lambda_value
            
        value_loss = v_lambda_value.mean()
        value_loss.backward(retain_graph=True)
        value_optimizer.step()  
        
        total_actor_reward = rewards_tensor.sum(dim=0)  # shape (B,)
        mean_actor_reward = total_actor_reward.mean()
        print(f"Recompensa acumulada média do ator: {mean_actor_reward.item()}")
        
        wandb.log({"Mean_Reward": mean_actor_reward, "ValueLoss": value_loss, "ActorLoss": actor_loss})
    
    return actor, value_net

def main():
    device = training_device()
    # Carrega o ambiente Cartpole Swingup do DM Control Suite
    env = suite.load(domain_name="cartpole", task_name="swingup", task_kwargs={'time_limit': 10})
    env = pixels.Wrapper(env, pixels_only=True,
                         render_kwargs={'height': 84, 'width': 84, 'camera_id': 0})
    
    # Define dimensões dos modelos
    batch_size = 2000
    obs_shape = (batch_size, 84, 84)
    action_dim = env.action_spec().shape[0]  # Geralmente 1 para cartpole
    hidden_dim = 300
    latent_dim = 300
    
    # Instancia os modelos e move para o device selecionado
    world_model = DreamerWorldModel(obs_shape, latent_dim, action_dim, hidden_dim).to(device)
    actor = ActionDecoder(action_dim, 3, 300, device=device)
    value_net = ValueNet(latent_dim).to(device)
    
    dummy_input = torch.randn(1, latent_dim, device=device)
    _ = actor(dummy_input)  # Executa forward para criar os parâmetros
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=8e-5)
    value_optimizer = torch.optim.Adam(value_net.parameters(), lr=8e-5)
    
    wandb.init(project="behavior", name="dgx-bh-6")
    
    for i in range(10):
        data_sequence = collect_replay_buffer(env, 15, ReplayBuffer())
        
        for iteration in range(100000):
            print(f"Iteração: {iteration}")
            horizon = 1
            # Executa uma iteração de behavior learning, passando o device
            actor, value_net = behavior_learning(
                data_sequence,
                world_model,
                batch=batch_size,
                horizon=horizon,
                actor=actor,
                value_net=value_net,
                device=device,
                lambda_=0.95,
                value_optimizer=value_optimizer,
                actor_optimizer=actor_optimizer
            )

if __name__ == '__main__':
    main()
