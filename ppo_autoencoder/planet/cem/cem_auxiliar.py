import gymnasium as gym
import numpy as np
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

import os # Importar para criar diretórios para vídeos
import matplotlib.pyplot as plt


def setup_env(env_id): # Removido run_name daqui, pois não é usado diretamente nas funções setup_*
    """
    Configura e retorna um ambiente Gymnasium com base no ID fornecido.
    Adiciona atributos personalizados 'type' e 'shape' ao action_space
    para compatibilidade com o código de políticas e CEM.

    Args:
        env_id (str): O ID do ambiente a ser criado (ex: "cartpole", "pendulum").

    Returns:
        tuple: Uma tupla contendo:
            - env (gym.Env): A instância do ambiente Gymnasium.
            - obs_shape (tuple/int): A dimensão do espaço de observação. Pode ser int para obs vetoriais ou tuple para imagens.
            - act_shape (int): A dimensão do espaço de ação (número de ações discretas ou dimensão do vetor de ação contínuo).
    """
    if env_id.lower() == "cartpole":
        return setup_cartpole()
    elif env_id.lower() == "pendulum":
        return setup_pendulum()
    else:
        # Passa env_id, mas 'run_name' e 'capture_video' não são necessários aqui
        # pois setup_atari é mais um "builder" de ambiente base.
        # A informação de vídeo e run_name é mais relevante na chamada evaluate_theta.
        return setup_atari(env_id=env_id)


def setup_atari(env_id, capture_video=False): # Removido run_name, pois não é usado aqui
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

    # Para ambientes de imagem como Atari, obs_shape será uma tupla (canais, altura, largura)
    # Por exemplo: (4, 84, 84). O CEM usará latent_dim em vez disso para theta_dim.
    obs_shape = env.observation_space.shape
    act_shape = env.action_space.n # Ações são discretas para Atari

    # Adiciona um atributo personalizado 'type' ao action_space para que a política possa identificá-lo
    env.action_space.type = "discrete" # Atari é sempre discreto

    return env, obs_shape, act_shape # Retorna a tupla esperada


def setup_pendulum():
    env = gym.make("Pendulum-v1")
    obs_shape = env.observation_space.shape[0]
    act_shape = env.action_space.shape[0]

    env.action_space.type = "continuous"
    return env, obs_shape, act_shape


def setup_cartpole():
    env = gym.make("CartPole-v1")
    obs_shape = env.observation_space.shape[0]
    act_shape = env.action_space.n

    env.action_space.type = "discrete"
    return env, obs_shape, act_shape


def plot_history(history, env_id, num_episodes, expt_time):
    f, ax = plt.subplots(nrows=2, sharex=True, figsize=(10, 10))
    f.suptitle(f"{env_id} {num_episodes} samples {expt_time:.0f} seconds") # Usando f-string
    ax[0].plot(history["epoch"], history["avg_rew"], label="population")
    ax[0].plot(history["epoch"], history["avg_elites"], label="elite")
    ax[0].legend()
    ax[0].set_ylabel("average rewards")

    ax[1].plot(history["epoch"], history["std_rew"], label="population")
    ax[1].plot(history["epoch"], history["std_elites"], label="elites")
    ax[1].legend()
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("standard deivation rewards")

    f.savefig(f"./{env_id}/learning.png") # Usando f-string


def setup_policy(env, theta, use_latent, latent_dim):
    if hasattr(env.action_space, 'type') and env.action_space.type == "discrete":
        return DiscretePolicy(env, theta, use_latent, latent_dim)
    elif isinstance(env.action_space, gym.spaces.Discrete):
        return DiscretePolicy(env, theta, use_latent, latent_dim)
    elif hasattr(env.action_space, 'type') and env.action_space.type == "continuous":
        return ContinuousPolicy(env, theta)
    elif isinstance(env.action_space, gym.spaces.Box):
        return ContinuousPolicy(env, theta)
    else:
        raise ValueError(f"Tipo de espaço de ação não suportado: {type(env.action_space)}")


class ContinuousPolicy:
    def __init__(self, env, theta):
        self.env = env
        obs_shape = env.observation_space.shape[0]
        act_shape = env.action_space.shape[0]
        expected_theta_len = (obs_shape + 1) * act_shape
        assert len(theta) == expected_theta_len, \
            f"Tamanho incorreto de theta para ContinuousPolicy. Esperado {expected_theta_len}, recebido {len(theta)}."

        self.parameter_dim = obs_shape * act_shape
        self.W = theta[: self.parameter_dim].reshape(obs_shape, act_shape)
        self.b = theta[self.parameter_dim :]

    def act(self, observation):
        action = observation.dot(self.W) + self.b
        return np.clip(action, self.env.action_space.low, self.env.action_space.high)

class DiscretePolicy:
    def __init__(self, env, theta, use_latent = False, latent_dim = 0):
        self.env = env
        if use_latent: # Se estiver usando latent_dim, use-o como a dimensão de observação
            obs_input_dim = latent_dim
        else: # Caso contrário, use a dimensão real do espaço de observação
            # Para ambientes de imagem (que têm obs_space.shape como tupla), precisamos
            # pegar o produto das dimensões se não usarmos o encoder.
            # Como estamos sempre usando o encoder para imagens aqui, essa parte
            # seria mais complexa sem o use_latent.
            # Assumimos que se use_latent é False, o env.observation_space.shape[0] faz sentido.
            obs_input_dim = env.observation_space.shape[0] 
            # Note: Para ambientes Atari sem encoder, obs_shape seria (4, 84, 84),
            # e obs_input_dim deveria ser 4*84*84. O `use_latent` simplifica isso.

        num_actions = env.action_space.n
        # A dimensão esperada do theta é (dimensão de entrada da política + 1 para o bias) * num_actions
        # O bias está embutido no theta_dim total, e 'b' é extraído posteriormente.
        # No seu setup, 'b' é pego de theta[parameter_dim:] o que significa que o bias é adicionado
        # separadamente e não é parte da matriz W.
        # Então, o cálculo do `expected_theta_len` deve ser `obs_input_dim * num_actions + num_actions`
        # ou `(obs_input_dim + 1) * num_actions`
        # De acordo com seu código original `self.b = theta[self.parameter_dim :]`,
        # o `self.b` tem `num_actions` elementos.
        # E `self.parameter_dim` é `obs_shape * num_actions`.
        # Assim, o comprimento total é `obs_input_dim * num_actions + num_actions`
        expected_theta_len = (obs_input_dim) * num_actions + num_actions # pesos + bias para cada ação
        
        assert len(theta) == expected_theta_len, \
            f"Tamanho incorreto de theta para DiscretePolicy. Esperado {expected_theta_len}, recebido {len(theta)}."

        self.parameter_dim = obs_input_dim * num_actions
        self.W = theta[: self.parameter_dim].reshape(obs_input_dim, num_actions)
        self.b = theta[self.parameter_dim :]

    def act(self, observation):
        y = observation.dot(self.W) + self.b
        action = np.argmax(y)
        return action