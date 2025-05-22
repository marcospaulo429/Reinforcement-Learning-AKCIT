import gymnasium as gym
import numpy as np # Embora numpy não seja diretamente usado aqui, é comum em projetos de RL


def setup_env(env_id):
    """
    Configura e retorna um ambiente Gymnasium com base no ID fornecido.
    Adiciona atributos personalizados 'type' e 'shape' ao action_space
    para compatibilidade com o código de políticas e CEM.

    Args:
        env_id (str): O ID do ambiente a ser criado (ex: "cartpole", "pendulum").

    Returns:
        tuple: Uma tupla contendo:
            - env (gym.Env): A instância do ambiente Gymnasium.
            - obs_shape (int): A dimensão do espaço de observação.
            - act_shape (int): A dimensão do espaço de ação (número de ações discretas ou dimensão do vetor de ação contínuo).
    """
    if env_id.lower() == "cartpole":
        return setup_cartpole()
    elif env_id.lower() == "pendulum":
        return setup_pendulum()
    else:
        return setup_atari()

import gymnasium as gym
import os # Importar para criar diretórios para vídeos

def setup_atari(env_id, capture_video=False, run_name="cem_atari"):
    """
    Configura e retorna um ambiente Gymnasium com base no ID fornecido.
    Aplica wrappers comuns e, opcionalmente, gravação de vídeo.

    Args:
        env_id (str): O ID do ambiente a ser criado (ex: "CartPole-v1", "Pendulum-v1").
        capture_video (bool, optional): Se True, o ambiente será envolvido com RecordVideo
                                        para gravar a execução. Defaults to False.
        run_name (str, optional): Um nome para a pasta de vídeos, se a gravação estiver ativada.
                                  Ajuda a organizar os vídeos. Defaults to "cem_run".

    Returns:
        tuple: Uma tupla contendo:
            - env (gym.Env): A instância do ambiente Gymnasium configurada.
            - obs_shape (int): A dimensão do espaço de observação.
            - act_shape (int): A dimensão do espaço de ação (número de ações discretas
                               ou dimensão do vetor de ação contínuo).
    """
    # 1. Cria o ambiente base.
    # render_mode="rgb_array" é necessário para RecordVideo. Se não for gravar, pode ser None.
    env = gym.make(env_id, render_mode="rgb_array" if capture_video else None)

    # 2. Aplica o wrapper de gravação de vídeo, se solicitado.
    if capture_video:
        # Cria o diretório para salvar os vídeos, se não existir.
        video_folder = f"videos/{run_name}"
        os.makedirs(video_folder, exist_ok=True)
        # Envolve o ambiente com RecordVideo.
        env = gym.wrappers.RecordVideo(env, video_folder=video_folder, name_prefix=env_id)

    # 3. Aplica o wrapper para registrar estatísticas do episódio (recompensa total, duração).
    # Isso é útil para o log e plotagem do histórico.
    env = gym.wrappers.RecordEpisodeStatistics(env)

    # 4. Determina obs_shape e act_shape.
    # obs_shape: A primeira dimensão do espaço de observação (número de características).
    obs_shape = env.observation_space.shape[0]

    # act_shape: Depende se o espaço de ação é discreto ou contínuo.
    if isinstance(env.action_space, gym.spaces.Discrete):
        # Para espaços discretos (ex: CartPole), act_shape é o número de ações possíveis.
        act_shape = env.action_space.n
    elif isinstance(env.action_space, gym.spaces.Box):
        # Para espaços contínuos (ex: Pendulum), act_shape é a dimensão do vetor de ação.
        act_shape = env.action_space.shape[0]
    else:
        # Levanta um erro se o tipo de espaço de ação não for reconhecido.
        raise ValueError(f"Tipo de espaço de ação não suportado: {type(env.action_space)}")

    return env, obs_shape, act_shape

# Nota: As funções setup_cartpole e setup_pendulum separadas não são mais necessárias
# pois setup_env agora lida com a criação de qualquer ambiente Gym válido.
# Você pode chamar setup_env("CartPole-v1") ou setup_env("Pendulum-v1") diretamente.

def setup_pendulum():
    """
    Configura o ambiente Pendulum-v1.

    Returns:
        tuple: (env, obs_shape, act_shape)
    """
    # Usando Pendulum-v1, que é a versão mais recente e recomendada
    env = gym.make("Pendulum-v1")
    obs_shape = env.observation_space.shape[0]
    act_shape = env.action_space.shape[0] # Para espaços de ação contínuos, é a dimensão do vetor

    # Adiciona um atributo personalizado 'type' ao action_space para que a política possa identificá-lo
    env.action_space.type = "continuous"
    return env, obs_shape, act_shape


def setup_cartpole():
    """
    Configura o ambiente CartPole-v1.

    Returns:
        tuple: (env, obs_shape, act_shape)
    """
    # Usando CartPole-v1, que é a versão mais recente e recomendada
    env = gym.make("CartPole-v1")
    obs_shape = env.observation_space.shape[0]
    act_shape = env.action_space.n # Para espaços de ação discretos, é o número de ações

    # Adiciona um atributo personalizado 'type' ao action_space para que a política possa identificá-lo
    env.action_space.type = "discrete"

    # O espaço de ação discreto no Gymnasium tem shape vazio (),
    # mas o código original espera (1,) para o cálculo de theta_dim.
    # Adicionamos este atributo personalizado para manter a compatibilidade.
    return env, obs_shape, act_shape


import matplotlib.pyplot as plt


def plot_history(history, env_id, num_episodes, expt_time):
    f, ax = plt.subplots(nrows=2, sharex=True, figsize=(10, 10))
    f.suptitle("{} {} samples {:0.0f} seconds".format(env_id, num_episodes, expt_time))
    ax[0].plot(history["epoch"], history["avg_rew"], label="population")
    ax[0].plot(history["epoch"], history["avg_elites"], label="elite")
    ax[0].legend()
    ax[0].set_ylabel("average rewards")

    ax[1].plot(history["epoch"], history["std_rew"], label="population")
    ax[1].plot(history["epoch"], history["std_elites"], label="elites")
    ax[1].legend()
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("standard deivation rewards")

    f.savefig("./{}/learning.png".format(env_id))

import gymnasium as gym
import numpy as np


def setup_policy(env, theta):
    """
    Cria uma instância da política apropriada (DiscretePolicy ou ContinuousPolicy)
    com base no tipo do espaço de ação do ambiente.

    Args:
        env (gym.Env): O ambiente Gymnasium.
        theta (np.ndarray): O vetor de parâmetros da política.

    Returns:
        object: Uma instância de DiscretePolicy ou ContinuousPolicy.

    Raises:
        ValueError: Se o tipo do espaço de ação não for discreto nem contínuo.
    """
    if hasattr(env.action_space, 'type') and env.action_space.type == "discrete":
        return DiscretePolicy(env, theta)
    elif isinstance(env.action_space, gym.spaces.Discrete):
        return DiscretePolicy(env, theta)
    elif hasattr(env.action_space, 'type') and env.action_space.type == "continuous":
        return ContinuousPolicy(env, theta)
    elif isinstance(env.action_space, gym.spaces.Box):
        return ContinuousPolicy(env, theta)
    else:
        raise ValueError(f"Tipo de espaço de ação não suportado: {type(env.action_space)}")


class ContinuousPolicy:
    """
    Uma política linear para espaços de ação contínuos.
    A ação é calculada como: action = clip(observation.dot(W) + b, low, high).
    """
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
    """
    Uma política linear para espaços de ação discretos.
    A ação é selecionada como o índice que maximiza a saída de uma função linear
    aplicada à observação, seguida pela adição de um bias.
    """
    def __init__(self, env, theta):
        """
        Inicializa a política discreta.

        Args:
            env (gym.Env): O ambiente Gymnasium associado. Usado para obter informações
                           sobre o espaço de observação e o espaço de ação.
            theta (np.ndarray): Um vetor 1D contendo os parâmetros da política.
                                Esses parâmetros serão usados para definir a matriz de
                                pesos (W) e o vetor de bias (b).
        """
        self.env = env
        # Obtém a dimensionalidade do espaço de observação.
        # Por exemplo, para o CartPole, obs_shape é 4.
        obs_shape = env.observation_space.shape[0]
        # Obtém o número de ações discretas possíveis no ambiente.
        # Por exemplo, para o CartPole, num_actions é 2 (esquerda ou direita).
        num_actions = env.action_space.n
        # Calcula o tamanho esperado do vetor de parâmetros 'theta'.
        # Para cada ação, temos 'obs_shape' pesos (um para cada dimensão da observação)
        # e 1 bias. Multiplicamos pelo número de ações para cobrir todas elas.
        expected_theta_len = (obs_shape + 1) * num_actions
        # Garante que o vetor de parâmetros 'theta' fornecido tem o tamanho esperado.
        # Se não tiver, isso indica que a política não foi inicializada corretamente.
        assert len(theta) == expected_theta_len, \
            f"Tamanho incorreto de theta para DiscretePolicy. Esperado {expected_theta_len}, recebido {len(theta)}."

        # Calcula a dimensão total dos pesos da matriz 'W'.
        # É o número de dimensões da observação multiplicado pelo número de ações.
        self.parameter_dim = obs_shape * num_actions
        # Extrai os pesos da matriz 'W' da primeira parte do vetor 'theta'.
        # A matriz 'W' tem uma forma de (obs_shape, num_actions), onde cada coluna
        # corresponde aos pesos para calcular a preferência por uma ação específica
        # com base na observação.
        self.W = theta[: self.parameter_dim].reshape(obs_shape, num_actions)
        # Extrai os biases da última parte do vetor 'theta'.
        # O vetor 'b' tem um tamanho igual ao número de ações, com um bias para cada ação.
        self.b = theta[self.parameter_dim :]

    def act(self, observation):
        """
        Decide qual ação tomar com base na observação atual.

        Args:
            observation (np.ndarray): A observação atual do ambiente.

        Returns:
            int: O índice da ação discreta a ser tomada.
        """
        # Calcula uma pontuação (ou "logit") para cada ação possível.
        # Isso é feito multiplicando a observação pela matriz de pesos 'W' e
        # adicionando o vetor de bias 'b'.
        # O resultado 'y' é um vetor onde cada elemento corresponde à pontuação de uma ação.
        y = observation.dot(self.W) + self.b
        # Seleciona a ação com a maior pontuação. 'np.argmax(y)' retorna o índice
        # do elemento máximo no vetor 'y'. Como os índices correspondem às ações
        # discretas, este índice é a ação escolhida pela política.
        action = np.argmax(y)
        return action