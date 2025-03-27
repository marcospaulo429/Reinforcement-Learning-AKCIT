class ReplayBuffer:
    def __init__(self):
        self.buffer = []

    def add_episode(self, episode_data):
        self.buffer.append(episode_data)