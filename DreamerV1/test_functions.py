import torch
import torch.nn.functional as F
import random
import numpy as np # Importar se ainda não estiver importado
from utils import test_world_model

# --- Mock para o SummaryWriter (ainda necessário para simular o log) ---
class MockSummaryWriter:
    def add_image(self, tag, img_tensor, global_step):
        # Apenas imprime para simular a adição da imagem
        print(f"MockSummaryWriter: Adicionando imagem '{tag}' no global_step {global_step} com shape {img_tensor.shape}")

    def add_scalar(self, tag, value, global_step):
        # Apenas imprime para simular a adição do escalar
        print(f"MockSummaryWriter: Adicionando escalar '{tag}' com valor {value} no global_step {global_step}")

# --- Mocks de "Modelos" usando Apenas Funções que Retornam Tensores ---

# Mock do Encoder: Recebe uma observação e retorna mu e logvar
def mock_encoder_func(obs_tensor, latent_dim=32):
    batch_size = obs_tensor.shape[0]
    mu = torch.randn(batch_size, latent_dim, device=obs_tensor.device)
    logvar = torch.randn(batch_size, latent_dim, device=obs_tensor.device)
    return mu, logvar

# Mock do Decoder: Recebe um sample latente e retorna uma imagem
def mock_decoder_func(latent_sample_tensor, obs_channels=1, obs_h=64, obs_w=64):
    batch_size = latent_sample_tensor.shape[0]
    # Retorna um tensor de imagem com as dimensões esperadas
    return torch.rand(batch_size, obs_channels, obs_h, obs_w, device=latent_sample_tensor.device)

# Mock do Transition Model: Recebe o estado anterior e ação, retorna o próximo estado
# Para simplificar, esta função mockará o método _transition diretamente
def mock_transition_func(prev_state, action_one_hot, belief_size=256, state_size=32):
    batch_size = prev_state['sample'].shape[0]
    device = action_one_hot.device

    # Simula a evolução do sample latente e do belief
    next_sample = torch.randn(batch_size, state_size, device=device)
    next_belief = torch.randn(batch_size, belief_size, device=device)
    
    return {
        'sample': next_sample,
        'rnn_state': torch.zeros(batch_size, belief_size, device=device), # Mantém como zero para simplicidade
        'mean': next_sample, # Para simplicidade, mean é o sample
        'stddev': torch.ones_like(next_sample), # Stddev é 1
        'belief': next_belief
    }

# Mock do Reward Model: Recebe belief e sample, retorna a recompensa
def mock_reward_func(belief_tensor, sample_tensor):
    batch_size = belief_tensor.shape[0]
    # Retorna um tensor de recompensa simulado
    return torch.rand(batch_size, 1, device=belief_tensor.device) # Recompensa entre 0 e 1

# Função reparameterize (mantida como está)
def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

# --- Classe para simular os argumentos (args) ---
class MockArgs:
    def __init__(self, horizon_to_imagine=10, action_dim=4, obs_channels=1, obs_h=64, obs_w=64, latent_dim=32, belief_size=256):
        self.horizon_to_imagine = horizon_to_imagine
        self.action_dim = action_dim
        self.obs_channels = obs_channels
        self.obs_h = obs_h
        self.obs_w = obs_w
        self.latent_dim = latent_dim # Adicionado para uso nos mocks
        self.belief_size = belief_size # Adicionado para uso nos mocks

# --- Executando o Teste ---
if __name__ == "__main__":
    print("\n--- Iniciando o script de teste para test_world_model (apenas tensores) ---")

    # Configurações para o teste
    global_step = 100
    writer = MockSummaryWriter()
    
    # Instanciando o args primeiro para definir os tamanhos corretos para os mocks
    args = MockArgs(horizon_to_imagine=5, action_dim=4, obs_channels=1, obs_h=64, obs_w=64, latent_dim=32, belief_size=256)

    # Passando funções (ou objetos com métodos) como mocks
    # Note que aqui passamos funções lambdas ou funções diretamente
    encoder = lambda obs: mock_encoder_func(obs, args.latent_dim)
    decoder = lambda latent_sample: mock_decoder_func(latent_sample, args.obs_channels, args.obs_h, args.obs_w)
    
    # Para o transition_model, como sua função original espera um objeto com ._transition,
    # criamos um objeto mock simples que redireciona para a nossa função
    class MockTransitionModelObj:
        def __init__(self, belief_size, state_size):
            self.belief_size = belief_size # Adiciona o atributo belief_size
            self.state_size = state_size
        def _transition(self, prev_state, action_one_hot):
            return mock_transition_func(prev_state, action_one_hot, self.belief_size, self.state_size)
            
    transition_model = MockTransitionModelObj(args.belief_size, args.latent_dim)
    
    reward_model = mock_reward_func

    # Criar dados de observação e ações de exemplo
    obs_seq = torch.randint(0, 256, (1, 5, args.obs_channels, args.obs_h, args.obs_w), dtype=torch.uint8)
    actions_seq = torch.randint(0, args.action_dim, (1, 5))

    device = torch.device("cpu") # Usar CPU para o teste

    # Chamar a função a ser testada
    test_world_model(
        global_step=global_step,
        writer=writer,
        encoder=encoder,
        decoder=decoder,
        transition_model=transition_model,
        reward_model=reward_model,
        obs_seq=obs_seq,
        actions_seq=actions_seq,
        args=args,
        device=device,
        reparameterize=reparameterize
    )

    print("\n--- Teste do script concluído. Verifique as saídas no console. ---")