from models import TransitionModel, RewardModel

from stable_baselines3.common.atari_wrappers import (  # isort:skip
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

def make_env(env_id, idx, capture_video, run_name):

    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)
        return env

    return thunk

"""
from dataclasses import dataclass
@dataclass
class Args:
    latent_dim: int = 512
    belief_size: int = 200
    hidden_size: int = 200
    action_dim: int = 4 # Exemplo: 4 ações para um ambiente Atari
    future_rnn: bool = True
    mean_only: bool = False
    min_stddev: float = 0.1
    num_layers: int = 2

# Instancia os argumentos
args = Args()

# Define o tamanho do batch para o teste
batch_size = 32

print("--- Testando TransitionModel ---")
# Instancia o TransitionModel
transition_model = TransitionModel(
    latent_dim=args.latent_dim,
    belief_size=args.belief_size,
    hidden_size=args.hidden_size,
    future_rnn=args.future_rnn,
    action_dim=args.action_dim,
    mean_only=args.mean_only,
    min_stddev=args.min_stddev,
    num_layers=args.num_layers
)

# Cria tensores de entrada simulados para _transition
prev_state_transition = {
    'sample': torch.randn(batch_size, args.latent_dim),
    'belief': torch.randn(batch_size, args.belief_size),
    'rnn_state': torch.randn(batch_size, args.belief_size)
}
prev_action_transition = torch.randn(batch_size, args.action_dim)

# Passa os tensores de entrada pelo _transition
transition_output = transition_model._transition(prev_state_transition, prev_action_transition)

# Imprime as dimensões de saída
print(f"Saída do TransitionModel._transition:")
for key, value in transition_output.items():
    print(f"  {key}: {value.shape}")

print("\n--- Testando RewardModel ---")
# Instancia o RewardModel
# Assumimos que hidden_dim para RewardModel é belief_size do TransitionModel
# e state_dim para RewardModel é latent_dim do TransitionModel.
# Isso significa que RewardModel espera (belief, sample) como entrada.
reward_model = RewardModel(
    hidden_dim=args.belief_size, # h de dimensão belief_size
    state_dim=args.latent_dim # s de dimensão latent_dim
)

# Cria tensores de entrada simulados para RewardModel
# h_reward seria o 'belief' do estado
# s_reward seria o 'sample' do estado
h_reward_input = torch.randn(batch_size, args.belief_size)
s_reward_input = torch.randn(batch_size, args.latent_dim)

# Passa os tensores de entrada pelo RewardModel
reward_output = reward_model.forward(h_reward_input, s_reward_input)

# Imprime as dimensões de saída
print(f"Saída do RewardModel.forward: {reward_output.shape}")

print("\n--- Testando TransitionModel._posterior ---")
# Cria tensor de observação simulado para _posterior
# Assumimos que a observação é codificada para latent_dim
obs_posterior = torch.randn(batch_size, args.latent_dim)

# Passa os tensores de entrada pelo _posterior
posterior_output = transition_model._posterior(prev_state_transition, prev_action_transition, obs_posterior)

# Imprime as dimensões de saída
print(f"Saída do TransitionModel._posterior:")
for key, value in posterior_output.items():
    print(f"  {key}: {value.shape}")
"""