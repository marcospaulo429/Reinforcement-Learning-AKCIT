import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from dm_control import suite
from DreamerV1.utils.replay_buffer import ReplayBuffer
from DreamerV1.algorithms.general_algo import collect_replay_buffer
from dm_control.suite.wrappers import pixels
from DreamerV1.algorithms.train_wm import DreamerWorldModel
from actor_critic import ValueNet
from DreamerV1.models.models import ActionDecoder
import wandb
from DreamerV1.utils.auxiliares import training_device

torch.autograd.set_detect_anomaly(True)

def compute_v(rewards, values, tau, k, gamma=0.99, lamb=0.95, t=0):
    """
    v_lambda : (1,B)
    """
    horizon, B = rewards.shape
    h = min(tau + k, t + horizon - 1)
    
    v = torch.zeros(1, B, dtype=rewards.dtype, device=rewards.device)
    
    for n in range(tau, h):
        v3 = gamma ** (n - tau)
        v4 = gamma ** (h - tau)
        v1 = v3 * rewards[n]
        v2 = v4 * values[h]
        v = v1 + v2 + v
        
    return v

def compute_v_lambda(rewards, values, tau, gamma=0.99, lamb=0.95):
    horizon, B = rewards.shape
    v_lambda = torch.zeros(1, B, dtype=rewards.dtype, device=rewards.device)
    
    for n in range(horizon):
        v1 = (lamb ** (n - 1) * compute_v(rewards, values, tau, n))
        v2 = (lamb ** (horizon - 1))
        v3 = compute_v(rewards, values, tau, horizon)
        v_lambda = v1 + v2 * v3 + v_lambda
    v_lambda = (1 - lamb) * v_lambda
    
    return v_lambda

def behavior_learning(
    data_sequence, world_model, batch, horizon,
    actor, value_net, device, lambda_=0.95,
    value_optimizer=None, actor_optimizer=None
):
    """
    Função de aprendizado que:
      1) Carrega dados reais (obs, actions).
      2) Realiza rollouts imaginados (com world_model) de tamanho 'horizon'.
      3) Calcula V_lambda para cada item do batch.
      4) Atualiza as redes ator e valor.
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
    
    # Cria tensores no device selecionado
    obs_tensor = torch.tensor(obs_array, dtype=torch.float32, device=device)
    action_tensor = torch.tensor(action_array, dtype=torch.float32, device=device)
    
    dataset = TensorDataset(obs_tensor, action_tensor)
    data_loader = DataLoader(dataset, batch_size=batch, shuffle=True)
        
    for batch_obs, batch_action in data_loader:
        batch_obs = batch_obs.to(device)
        batch_action = batch_action.to(device)
        
        batch_size = batch_obs.size(0)
        hidden_dim = world_model.transition_model.gru.hidden_size
        
        # Estado oculto inicial
        prev_hidden = torch.zeros(batch_size, hidden_dim, device=device)
        
        # Passo 0: execução real
        latent, hidden, mean, std, reward, recon_obs = world_model(batch_obs, batch_action, prev_hidden)
        
        # Armazena rewards e values
        rewards = [reward.detach().view(-1)]
        values = [value_net(latent.detach()).view(-1)]
                
        # Rollout imaginado
        current_obs = recon_obs
        for i in range(horizon):
            current_action = actor(latent).rsample()
            latent, hidden, mean, std, reward, recon_obs = world_model(current_obs, current_action, hidden)
            rewards.append(reward.detach().view(-1))
            values.append(value_net(latent.detach()).view(-1))
            current_obs = recon_obs
        
        # Empilha para formar tensores (T, B)
        rewards_tensor = torch.stack(rewards, dim=0)
        values_tensor  = torch.stack(values, dim=0)
        
        # Cálculo para o ator
        v_lambda_actor = torch.zeros(1, batch_size, dtype=rewards_tensor.dtype, device=rewards_tensor.device, requires_grad=True)
        for tau in range(horizon):
            v_lam_tau = compute_v_lambda(rewards_tensor[tau:], values_tensor[tau:], tau=0, gamma=0.99, lamb=0.95)
            v_lambda_actor = + v_lam_tau + v_lambda_actor
        
        actor_loss = torch.mean(-v_lambda_actor)
        actor_loss.backward(retain_graph=True)
        actor_optimizer.step()
                
        # Cálculo para o valor
        v_lambda_value = torch.zeros(1, batch_size, dtype=rewards_tensor.dtype, device=rewards_tensor.device, requires_grad=True)
        for tau in range(horizon):
            v_lam_tau = compute_v_lambda(rewards_tensor[tau:], values_tensor[tau:], tau=0, gamma=0.99, lamb=0.95)
            v_lambda_value = (((values_tensor[tau] - v_lam_tau) ** 2) / 2) + v_lambda_value
            
        value_loss = torch.mean(v_lambda_value)
        value_loss.backward(retain_graph=True)
        value_optimizer.step()  
        
        mean_actor_reward = torch.mean(rewards_tensor)
        print(actor_loss,value_loss,mean_actor_reward)
        wandb.log({"Mean_Reward": mean_actor_reward, "ValueLoss": value_loss, "ActorLoss": actor_loss})
    
    return actor, value_net

def main():
    device = training_device()
    # Carrega o ambiente Cartpole Swingup do DM Control Suite
    env = suite.load(domain_name="cartpole", task_name="swingup", task_kwargs={'time_limit': 10})
    env = pixels.Wrapper(env, pixels_only=True,
                         render_kwargs={'height': 84, 'width': 84, 'camera_id': 0})
    
    # Define dimensões dos modelos
    batch_size = 1500
    obs_shape = (batch_size, 84, 84)
    action_dim = env.action_spec().shape[0]
    hidden_dim = 200
    latent_dim = 200
    
    # Instancia os modelos e move para o device
    world_model = DreamerWorldModel(obs_shape, latent_dim, action_dim, hidden_dim).to(device)
    actor = ActionDecoder(action_dim, 3, 200, device=device)
    value_net = ValueNet(latent_dim).to(device)
    
    dummy_input = torch.randn(1, latent_dim, device=device)
    _ = actor(dummy_input)  # Executa forward para criar os parâmetros
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=8e-5)
    value_optimizer = torch.optim.Adam(value_net.parameters(), lr=8e-5)
    
    wandb.init(project="behavior", name="m1-bh-8")
    
    data_sequence = collect_replay_buffer(env, 12, ReplayBuffer())
    
    for iteration in range(1000):
        print(f"\n================ Iteração: {iteration} =================")
        horizon = 1
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
