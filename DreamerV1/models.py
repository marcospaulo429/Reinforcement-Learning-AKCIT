import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent, TransformedDistribution, TanhTransform, Bernoulli, Categorical
import numpy as np

# Base auxiliar similar à tools.Module para criação preguiçosa de camadas.
class Module(nn.Module):
    def __init__(self, device =None):
        super().__init__()
        self.device = device
        self._layers = {}

    def get(self, name, module_class, *args, **kwargs):
        if name not in self._layers:
            # Para camadas lineares, usamos o LazyLinear para inferência dinâmica do in_features.
            if module_class == nn.Linear:
                self._layers[name] = nn.LazyLinear(*args, **kwargs)
            else:
                self._layers[name] = module_class(*args, **kwargs)
                
            if self.device is not None:
                self._layers[name] = self._layers[name].to(self.device)
                
            self.add_module(name, self._layers[name])
        return self._layers[name]

# Implementação de um static_scan simples para iterar sobre a dimensão temporal.
def static_scan(fn, sequence, initial):
    """
    Aplica fn recursivamente sobre a sequência.
    - sequence: tupla de tensores com formato (time, batch, ...).
    - initial: estado inicial (pode ser um dicionário ou outro objeto).
    Retorna (estado_final, out) onde out é um dicionário com os outputs empilhados ao longo do tempo.
    """
    outputs = []
    state = initial
    time_steps = sequence[0].size(0)
    for t in range(time_steps):
        # Para cada t, pegamos os inputs correspondentes de cada tensor da sequência.
        inputs_t = [seq[t] for seq in sequence]
        state, out = fn(state, inputs_t)
        outputs.append(out)
    # Reempilha cada chave dos outputs ao longo da dimensão tempo.
    out_dict = {}
    for key in outputs[0].keys():
        out_dict[key] = torch.stack([o[key] for o in outputs], dim=0)
    return state, out_dict

# Wrapper para encapsular distribuições com sample e log_prob.
class SampleDist:
    def __init__(self, dist):
        self.dist = dist

    def sample(self):
        # Usamos rsample() se disponível para permitir gradientes.
        return self.dist.rsample() if hasattr(self.dist, "rsample") else self.dist.sample()

    def log_prob(self, value):
        return self.dist.log_prob(value)

    def __getattr__(self, name):
        return getattr(self.dist, name)

# Distribuição para variáveis categóricas one-hot.
class OneHotDist(Categorical):
    def sample(self):
        sample = super().sample()
        return F.one_hot(sample, num_classes=self.probs.size(-1)).float()

    def log_prob(self, value):
        # Converte a codificação one-hot para índices.
        indices = value.argmax(dim=-1)
        return super().log_prob(indices)

###########################################
# Modelo RSSM
###########################################
class RSSM(Module):
    def __init__(self, stoch=30, deter=200, hidden=200, act=F.elu):
        super().__init__()
        self._activation = act
        self._stoch_size = stoch
        self._deter_size = deter
        self._hidden_size = hidden
        # A implementação assume que a entrada para o GRUCell terá dimensão _deter_size.
        self._cell = nn.GRUCell(self._deter_size, self._deter_size)

    def initial(self, batch_size, device):
        # Inicializa os estados: mean, std, stoch e o estado determinístico (hidden state do GRUCell).
        mean = torch.zeros(batch_size, self._stoch_size, device=device)
        std = torch.zeros(batch_size, self._stoch_size, device=device)
        stoch = torch.zeros(batch_size, self._stoch_size, device=device)
        deter = torch.zeros(batch_size, self._deter_size, device=device)
        return {'mean': mean, 'std': std, 'stoch': stoch, 'deter': deter}

    def observe(self, embed, action, state=None):
        # Assume que embed e action possuem formato (batch, time, features).
        if state is None:
            state = self.initial(action.size(0), action.device)
        # Transpõe para (time, batch, features) para iterar no tempo.
        embed = embed.transpose(0, 1)
        action = action.transpose(0, 1)

        def step_fn(prev_state, inputs):
            act_t, emb_t = inputs
            post, prior = self.obs_step(prev_state, act_t, emb_t)
            # Atualiza o estado com o prior.
            return prior, {'post': post, 'prior': prior}

        _, outs = static_scan(step_fn, (action, embed), state)
        # Transpõe os outputs de volta para (batch, time, features).
        post = {k: v.transpose(0, 1) for k, v in outs['post'].items()}
        prior = {k: v.transpose(0, 1) for k, v in outs['prior'].items()}
        return post, prior

    def imagine(self, action, state=None):
        # Imagina trajetórias sem observações (apenas usando ações).
        if state is None:
            state = self.initial(action.size(0), action.device)
        action = action.transpose(0, 1)

        def step_fn(prev_state, inputs):
            a_t = inputs[0]
            next_state = self.img_step(prev_state, a_t)
            return next_state, next_state

        _, prior = static_scan(step_fn, (action,), state)
        prior = {k: v.transpose(0, 1) for k, v in prior.items()}
        return prior

    def get_feat(self, state):
        return torch.cat([state['stoch'], state['deter']], dim=-1)

    def get_dist(self, state):
        return Independent(Normal(state['mean'], state['std']), 1)

    def obs_step(self, prev_state, prev_action, embed):
        prior = self.img_step(prev_state, prev_action)
        x = torch.cat([prior['deter'], embed], dim=-1)
        x = self.get('obs1', nn.Linear, self._hidden_size)(x)
        x = self._activation(x)
        x = self.get('obs2', nn.Linear, 2 * self._stoch_size)(x)
        mean, std = torch.chunk(x, 2, dim=-1)
        std = F.softplus(std) + 0.1
        dist = Independent(Normal(mean, std), 1)
        stoch = dist.rsample()
        post = {'mean': mean, 'std': std, 'stoch': stoch, 'deter': prior['deter']}
        return post, prior

    def img_step(self, prev_state, prev_action):
        x = torch.cat([prev_state['stoch'], prev_action], dim=-1)
        x = self.get('img1', nn.Linear, self._hidden_size)(x)
        x = self._activation(x)
        # GRUCell: input x e hidden state prev_state['deter']
        deter = self._cell(x, prev_state['deter'])
        x = self.get('img2', nn.Linear, self._hidden_size)(deter)
        x = self._activation(x)
        x = self.get('img3', nn.Linear, 2 * self._stoch_size)(x)
        mean, std = torch.chunk(x, 2, dim=-1)
        std = F.softplus(std) + 0.1
        dist = Independent(Normal(mean, std), 1)
        stoch = dist.rsample()
        prior = {'mean': mean, 'std': std, 'stoch': stoch, 'deter': deter}
        return prior

###########################################
# ConvEncoder
###########################################
class ConvEncoder(Module):
    def __init__(self, depth=32, act=F.relu):
        super().__init__()
        self._act = act
        self._depth = depth
        # Não definimos as camadas aqui porque usamos o método get para criação preguiçosa.
        # Assume-se que a imagem de entrada possui 3 canais (RGB).

    def forward(self, obs):
        # obs é um dicionário contendo 'image' com formato (batch, C, H, W).
        x = obs
        # Aplicação de camadas convolucionais com stride 2.
        x = self.get('h1', nn.Conv2d, 3, 1 * self._depth, kernel_size=4, stride=2)(x)
        x = self._act(x)
        x = self.get('h2', nn.Conv2d, 1 * self._depth, 2 * self._depth, kernel_size=4, stride=2)(x)
        x = self._act(x)
        x = self.get('h3', nn.Conv2d, 2 * self._depth, 4 * self._depth, kernel_size=4, stride=2)(x)
        x = self._act(x)
        x = self.get('h4', nn.Conv2d, 4 * self._depth, 8 * self._depth, kernel_size=4, stride=2)(x)
        x = self._act(x)
        # Achata as dimensões espaciais.
        x = x.view(x.size(0), -1)
        return x

###########################################
# ConvDecoder
###########################################
class ConvDecoder(Module):
    def __init__(self, depth=32, act=F.relu, shape=(64, 64, 3)):
        super().__init__()
        self._act = act
        self._depth = depth
        self._shape = shape  # (H, W, C) – observe que em PyTorch a ordem geralmente é (C, H, W)
        self._out_channels = shape[2]
        self._out_size = (shape[0], shape[1])

    def forward(self, features):
        x = features
        x = self.get('h1', nn.Linear, 32 * self._depth)(x)
        x = x.view(-1, 32 * self._depth, 1, 1)
        x = self.get('h2', nn.ConvTranspose2d, 32 * self._depth, 4 * self._depth, kernel_size=5, stride=2)(x)
        x = self._act(x)
        x = self.get('h3', nn.ConvTranspose2d, 4 * self._depth, 2 * self._depth, kernel_size=5, stride=2)(x)
        x = self._act(x)
        x = self.get('h4', nn.ConvTranspose2d, 2 * self._depth, 1 * self._depth, kernel_size=6, stride=2)(x)
        x = self._act(x)
        x = self.get('h5', nn.ConvTranspose2d, 1 * self._depth, self._out_channels, kernel_size=6, stride=2)(x)
        # Ajusta para (batch, C, H, W)
        mean = x.view(x.size(0), self._out_channels, self._out_size[0], self._out_size[1])
        # Cria uma distribuição Normal com desvio padrão fixo igual a 1.
        dist = Independent(Normal(mean, torch.ones_like(mean)), len(self._out_size) + 1)
        return dist

###########################################
# DenseDecoder
###########################################
class DenseDecoder(Module):
    def __init__(self, shape, layers, units, dist='normal', act=F.elu):
        super().__init__()
        self._shape = shape
        self._layers = layers
        self._units = units
        self._dist = dist
        self._act = act

    def forward(self, features):
        x = features
        for index in range(self._layers):
            x = self.get(f'h{index}', nn.Linear, self._units)(x)
            x = self._act(x)
        x = self.get('hout', nn.Linear, int(np.prod(self._shape)))(x)
        x = x.view(x.size(0), *self._shape)
        if self._dist == 'normal':
            dist = Independent(Normal(x, torch.ones_like(x)), len(self._shape))
            return dist
        elif self._dist == 'binary':
            dist = Bernoulli(logits=x)
            dist = Independent(dist, len(self._shape))
            return dist
        else:
            raise NotImplementedError(self._dist)

###########################################
# ActionDecoder
###########################################
class ActionDecoder(Module):
    def __init__(self, size, num_layers, units, dist='tanh_normal', act=F.elu, device=None,
                 min_std=1e-4, init_std=5, mean_scale=5):
        super().__init__(device=device)
        self._size = size
        self.num_layers = num_layers   # Número de camadas
        self._units = units
        self._dist = dist
        self._act = act
        self._min_std = min_std
        self._init_std = init_std
        self._mean_scale = mean_scale
        self._raw_init_std = np.log(np.exp(self._init_std) - 1)

    def forward(self, features):
        x = features
        for index in range(self.num_layers):
            # A cada iteração, a camada é criada se ainda não existir
            x = self.get(f'h{index}', nn.Linear, self._units)(x)
            x = self._act(x)
        if self._dist == 'tanh_normal':
            x = self.get('hout', nn.Linear, 2 * self._size)(x)
            mean, std = torch.chunk(x, 2, dim=-1)
            mean = self._mean_scale * torch.tanh(mean / self._mean_scale)
            std = F.softplus(std + self._raw_init_std) + self._min_std
            base_dist = Normal(mean, std)
            # Aplica a transformação tanh.
            dist = TransformedDistribution(base_dist, TanhTransform(cache_size=1))
            dist = Independent(dist, 1)
            return dist
        elif self._dist == 'onehot':
            x = self.get('hout', nn.Linear, self._size)(x)
            # Aqui você pode definir o comportamento para a distribuição onehot.
            return x
        else:
            raise NotImplementedError(self._dist)
"""

### RSSM (Recurrent State-Space Model)

- **O que faz:**  
  Modela a dinâmica do ambiente combinando estados determinísticos (via GRU) e estados estocásticos. É usado para inferência de estados (posterior e prior) tanto com observações quanto “imaginando” trajetórias (sem observação).

- **Inputs:**  
  - **observe:**  
    - *embed:* Tensor de embeddings extraídos das observações, com forma (batch, tempo, dimensões do embedding).  
    - *action:* Tensor de ações, com forma (batch, tempo, dimensões da ação).  
  - **imagine:**  
    - *action:* Tensor de ações, com forma (batch, tempo, dimensões da ação).

- **Outputs:**  
  - **observe:**  
    - *post:* Dicionário com estados pós-observação contendo chaves: `mean`, `std`, `stoch`, `deter` – cada tensor tem forma (batch, tempo, dimensão correspondente).  
    - *prior:* Dicionário similar com estados “antecipados” (antes de observar o novo dado).  
  - **imagine:**  
    - *prior:* Dicionário de estados gerados pela dinâmica interna do modelo.

- **Métodos auxiliares:**  
  - `obs_step`: Combina o estado previsto com o embedding da observação para calcular o estado posterior.  
  - `img_step`: Atualiza o estado determinístico e amostra a parte estocástica baseado na ação anterior.

---

### ConvEncoder

- **O que faz:**  
  Converte imagens de entrada em uma representação latente (vetor de características) através de camadas convolucionais.

- **Inputs:**  
  - *obs:* Dicionário com a chave `image`, contendo imagens em formato (batch, C, H, W).

- **Outputs:**  
  - *features:* Tensor unidimensional para cada imagem (batch, feature_dim) resultante do achatamento da saída das convoluções.

---

### ConvDecoder

- **O que faz:**  
  Decodifica os vetores de características latentes em imagens reconstruídas utilizando uma camada densa seguida por camadas de convolução transposta.

- **Inputs:**  
  - *features:* Tensor com representação latente (batch, feature_dim).

- **Outputs:**  
  - Retorna uma distribuição (do tipo Independent Normal) sobre imagens, onde o parâmetro “mean” tem a forma (batch, C, H, W).  
  - Essa distribuição pode ser usada para amostrar imagens reconstruídas.

---

### DenseDecoder

- **O que faz:**  
  Reconstrói a saída (por exemplo, uma imagem ou outro sinal) a partir do vetor latente através de camadas densas (fully-connected).

- **Inputs:**  
  - *features:* Vetor latente (batch, feature_dim).

- **Outputs:**  
  - Retorna uma distribuição sobre a saída reconstruída:  
    - **Se `dist='normal'`:** Uma distribuição Independent Normal com média dada pela saída da rede e desvio padrão fixo (por exemplo, 1).  
    - **Se `dist='binary'`:** Uma distribuição Independent Bernoulli (usada para dados binários).

---

### ActionDecoder

- **O que faz:**  
  Mapeia o vetor latente para uma distribuição sobre ações. Pode gerar ações contínuas (usando uma transformação tanh de uma Normal) ou ações discretas (one-hot).

- **Inputs:**  
  - *features:* Vetor latente (batch, feature_dim).

- **Outputs:**  
  - Retorna uma distribuição para ações:  
    - **Se `dist='tanh_normal'`:** Uma distribuição onde a média é escalada e limitada por uma função tanh, e o desvio padrão é ajustado (após aplicar softplus e adicionar um mínimo).  
    - **Se `dist='onehot'`:** Uma distribuição categórica que gera ações em codificação one-hot.

---

Cada classe foi adaptada para funcionar com PyTorch, mantendo a lógica original dos modelos em TensorFlow, com o uso de camadas dinâmicas e funções específicas para amostragem e propagação de estados em séries temporais.
"""