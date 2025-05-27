import numpy as np
import torch
from collections import deque
import random

class EpisodeBuffer:
    def __init__(self, capacity, obs_shape, action_dim, num_envs=1):
        """
        Inicializa o buffer de armazenamento de episódios
        
        Args:
            capacity: Número máximo de episódios que o buffer pode armazenar
            obs_shape: Formato das observações (tuple)
            action_dim: Dimensão das ações
            num_envs: Número de ambientes paralelos (default=1)
        """
        self.capacity = capacity
        self.num_envs = num_envs
        self.buffer = deque(maxlen=capacity)
        self.episode_id = 0
        
        # Estrutura para armazenar episódios ativos (ainda não terminados)
        self.active_episodes = [{
            'obs': [],
            'actions': [],
            'rewards': [],
            'next_obs': [],
            'dones': []
        } for _ in range(num_envs)]
    
    def _create_empty_episode(self):
        """Cria um episódio vazio"""
        return {
            'obs': [],
            'actions': [],
            'rewards': [],
            'next_obs': [],
            'dones': []
        }
    
    def add_step(self, obs, action, reward, next_obs, done, env_idx=0):
        """
        Adiciona um passo ao episódio atual
        
        Args:
            obs: Observação atual
            action: Ação tomada
            reward: Recompensa recebida
            next_obs: Próxima observação
            done: Se o episódio terminou
            env_idx: Índice do ambiente (para multi-env)
        """
        # Converte tensores para numpy se necessário
        obs = obs.cpu().numpy() if torch.is_tensor(obs) else obs
        action = action.cpu().numpy() if torch.is_tensor(action) else action
        next_obs = next_obs.cpu().numpy() if torch.is_tensor(next_obs) else next_obs
        
        # Adiciona o passo ao episódio ativo
        self.active_episodes[env_idx]['obs'].append(obs)
        self.active_episodes[env_idx]['actions'].append(action)
        self.active_episodes[env_idx]['rewards'].append(reward)
        self.active_episodes[env_idx]['next_obs'].append(next_obs)
        self.active_episodes[env_idx]['dones'].append(done)
        
        # Se o episódio terminou, adiciona ao buffer e inicia novo episódio
        if done:
            self.finalize_episode(env_idx)
    
    def finalize_episode(self, env_idx=0):
        """
        Finaliza o episódio atual e adiciona ao buffer
        
        Args:
            env_idx: Índice do ambiente (para multi-env)
        """
        if len(self.active_episodes[env_idx]['obs']) == 0:
            return  # Nada para finalizar
        
        # Converte listas para arrays numpy
        episode = {
            'obs': np.array(self.active_episodes[env_idx]['obs']),
            'actions': np.array(self.active_episodes[env_idx]['actions']),
            'rewards': np.array(self.active_episodes[env_idx]['rewards']),
            'next_obs': np.array(self.active_episodes[env_idx]['next_obs']),
            'dones': np.array(self.active_episodes[env_idx]['dones']),
            'id': self.episode_id
        }
        self.episode_id += 1
        
        # Adiciona ao buffer (o deque automaticamente remove os mais antigos quando atinge capacidade)
        self.buffer.append(episode)
        
        # Reinicia o episódio ativo
        self.active_episodes[env_idx] = self._create_empty_episode()
    
    def sample_episodes(self, batch_size):
        """
        Amostra episódios aleatórios do buffer
        
        Args:
            batch_size: Número de episódios para amostrar
            
        Returns:
            Lista de episódios amostrados
        """
        if len(self.buffer) < batch_size:
            return random.sample(list(self.buffer), len(self.buffer))
        return random.sample(list(self.buffer), batch_size)
    
    def sample_steps(self, batch_size, max_seq_len=None):
        """
        Amostra passos aleatórios de episódios aleatórios
        
        Args:
            batch_size: Número de passos para amostrar
            max_seq_len: Comprimento máximo de sequência (para treino recorrente)
            
        Returns:
            Dicionário com batches de passos
        """
        steps = {
            'obs': [],
            'actions': [],
            'rewards': [],
            'next_obs': [],
            'dones': []
        }
        
        while len(steps['obs']) < batch_size:
            # Escolhe um episódio aleatório
            episode = random.choice(self.buffer)
            
            # Escolhe um ponto de início aleatório no episódio
            if max_seq_len is not None:
                start_idx = random.randint(0, max(0, len(episode['obs']) - max_seq_len))
                end_idx = start_idx + min(max_seq_len, len(episode['obs']) - start_idx)
            else:
                start_idx = 0
                end_idx = len(episode['obs'])
            
            # Adiciona os passos selecionados
            steps['obs'].extend(episode['obs'][start_idx:end_idx])
            steps['actions'].extend(episode['actions'][start_idx:end_idx])
            steps['rewards'].extend(episode['rewards'][start_idx:end_idx])
            steps['next_obs'].extend(episode['next_obs'][start_idx:end_idx])
            steps['dones'].extend(episode['dones'][start_idx:end_idx])
        
        # Corta para o tamanho exato do batch
        for k in steps:
            steps[k] = steps[k][:batch_size]
        
        # Converte para tensores
        return {
            'obs': torch.FloatTensor(np.array(steps['obs'])),
            'actions': torch.FloatTensor(np.array(steps['actions'])),
            'rewards': torch.FloatTensor(np.array(steps['rewards'])),
            'next_obs': torch.FloatTensor(np.array(steps['next_obs'])),
            'dones': torch.FloatTensor(np.array(steps['dones']))
        }
    
    def __len__(self):
        """Retorna o número de episódios armazenados"""
        return len(self.buffer)
    
