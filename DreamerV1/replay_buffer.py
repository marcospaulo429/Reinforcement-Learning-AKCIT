import csv
import json
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