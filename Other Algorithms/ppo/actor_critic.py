import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Configurar dispositivo (altere para o ID da GPU desejado se necessário)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Policy(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Policy, self).__init__()
        self.layer1 = nn.Linear(in_dim, 32)
        self.layer2 = nn.Linear(32, 16)
        self.layer3 = nn.Linear(16, out_dim)
        self.to(device)
        
    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32, device=device)
        x = torch.relu(self.layer1(obs))
        x = torch.relu(self.layer2(x))
        logits = self.layer3(x)
        action_probs = torch.softmax(logits, dim=-1)
        action_dist = torch.distributions.Categorical(action_probs)
        return action_probs, action_dist

class Critic(nn.Module):
    def __init__(self, in_dim):
        super(Critic, self).__init__()
        self.layer1 = nn.Linear(in_dim, 32)
        self.layer2 = nn.Linear(32, 16)
        self.layer3 = nn.Linear(16, 1)
        self.to(device)
        
    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32, device=device)
        x = torch.relu(self.layer1(obs))
        x = torch.relu(self.layer2(x))
        return self.layer3(x)

# Test code
if __name__ == "__main__":
    # Test configuration
    obs_dim = 4
    act_dim = 2
    
    # Initialize networks
    policy = Policy(in_dim=obs_dim, out_dim=act_dim)
    critic = Critic(in_dim=obs_dim)
    
    # Test data
    test_obs_np = np.random.rand(obs_dim)
    test_obs_tensor = torch.rand(obs_dim)
    
    # Test Policy
    print("=== Policy Test ===")
    print("Input (numpy):", test_obs_np)
    probs_np, dist_np = policy(test_obs_np)
    print("Output probs:", probs_np)
    print("Sample action:", dist_np.sample().item())
    print("Sum of probs:", torch.sum(probs_np).item())
    action = dist_np.sample()
    log_prob = dist_np.log_prob(action)
    print(action,log_prob)
    
    # Test Critic
    print("\n=== Critic Test ===")
    print("Input (numpy):", test_obs_np)
    value_np = critic(test_obs_np)
    print("Output value:", value_np)
    
    # Batch test
    print("\n=== Batch Test ===")
    batch_obs = torch.rand(5, obs_dim)
    batch_probs, batch_dist = policy(batch_obs)
    batch_values = critic(batch_obs)
    
    print("Batch probs shape:", batch_probs.shape)
    print("Batch values shape:", batch_values.shape)