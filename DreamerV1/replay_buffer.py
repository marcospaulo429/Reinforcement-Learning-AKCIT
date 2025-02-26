import random
import numpy as np
from auxiliares import converter_cinza

class ReplayBuffer:
    def __init__(self):
        self.buffer = []

    def add_episode(self, episode_data):
        self.buffer.append(episode_data)

    def save_to_csv(self, filename="replay_buffer.csv"): #TODO
        print()
        
    def load_from_csv(self, filename="replay_buffer.csv"): #TODO
        print()

def collect_replay_buffer(env,S,replay_buffer): 
    for i in range(S):
        print(f"Collecting episode {i+1}")
        time_step = env.reset()
        done = False
        episode_data = []
        obs_atual = converter_cinza(time_step.observation['pixels'])
        obs_atual = obs_atual.astype(np.float32) / 127.5 - 1.0
        while not done:
            action_spec = env.action_spec()
            random_action = np.random.uniform(low=action_spec.minimum,
                                              high=action_spec.maximum,
                                              size=action_spec.shape)
            time_step = env.step(random_action)
            done = time_step.last()
            obs_prox = converter_cinza(time_step.observation['pixels'])
            obs_prox = obs_prox.astype(np.float32) / 127.5 - 1.0
            step_data = {
                "obs": obs_atual,
                "action": random_action,
                "reward": time_step.reward if time_step.reward is not None else 0.0,
                "next_obs": obs_prox,
                "done": done
            }
            episode_data.append(step_data)
            obs_atual = obs_prox
        replay_buffer.add_episode(episode_data)
    print(f"Collected {S} episodes.") 
    return replay_buffer

def sample_data_sequences(replay_buffer, num_sequences, sequence_length):
    """
    Seleciona num_sequences (X) sequências, cada uma com sequence_length (Y) passos, 
    a partir dos episódios presentes em replay_buffer.
    
    Retorna um novo ReplayBuffer contendo essas sequências como episódios.
    """
    new_buffer = ReplayBuffer()
    
    # Filtra apenas episódios com comprimento suficiente
    valid_episodes = [ep for ep in replay_buffer.buffer if len(ep) >= sequence_length]
    if len(valid_episodes) == 0:
        raise ValueError("Nenhum episódio possui comprimento >= sequence_length")
    
    for _ in range(num_sequences):
        # Escolhe aleatoriamente um episódio entre os válidos
        episode = random.choice(valid_episodes)
        # Escolhe um índice de início aleatório para extrair uma sequência de tamanho sequence_length
        start_idx = random.randint(0, len(episode) - sequence_length)
        sub_episode = episode[start_idx : start_idx + sequence_length]
        new_buffer.add_episode(sub_episode)
        
    return new_buffer