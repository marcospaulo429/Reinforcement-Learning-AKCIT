import pandas as pd
import numpy as np
import random
from collections import defaultdict
from itertools import permutations

Q = defaultdict(float)

total_return = defaultdict(float)

N = defaultdict(int)

#Inicializando a política e a value function:
TAMANHO_QUADRADO = 3
ACOES = (0,1,2,3,4,5,6,7,8)

# Parâmetros
EPSILON = 0.5  # Probabilidade de exploração

def done(estado):
    if(estado[0]== 1 and estado[1]==1 and estado[2]==1 or
        estado[3]== 1 and estado[4]==1 and estado[5]==1 or
        estado[6]==1 and estado[7]==1 and estado[8]==1 or
        estado[0]==1 and estado[3]==1 and estado[6]==1 or
        estado[2]==1 and estado[4]==1 and estado[7]==1 or
        estado[2]==1 and estado[5]==1 and estado[8]==1 or
        estado[0]==1 and estado[4]==1 and estado[8]==1 or
        estado[2]==1 and estado[4]==1 and estado[6]==1):
        return True, 1

    elif(estado[0]==2 and estado[1]==2 and estado[2]==2 or
        estado[3]==2 and estado[4]==2 and estado[5]==2 or
        estado[6]==2 and estado[7]==2 and estado[8]==2  or
        estado[0]==2 and estado[3]==2 and estado[6]==2 or
        estado[2]==2 and estado[4]==2 and estado[7]==2 or
        estado[2]==2 and estado[5]==2 and estado[8]==2 or
        estado[0]==2 and estado[4]==2 and estado[8]==2 or
        estado[2]==2 and estado[4]==2 and estado[6]==2):

        return True, 2

    else:
        return False
    
def deu_velha(estado):
    if done(estado) is False and 0 not in estado:
        return True
    else:
        pass
            

def novo_estado(estado, acao):
    estado2 = estado.copy()
    estado2[acao] = 1
    return estado2

jogadas_que_ganham = np.array([[0,1,2],
                                   [3,4,5],
                                   [6,7,8],
                                   [0,3,6],
                                   [2,4,7],
                                   [2,5,8],
                                   [0,4,8],
                                   [2,4,6]])

todas_combinacoes = []

# Percorra cada lista dentro do array
for lista in jogadas_que_ganham:
    # Gere todas as permutações da lista atual
    permutacoes = list(permutations(lista))
    
    # Adicione as permutações à lista de todas as combinações
    todas_combinacoes.extend(permutacoes)

###----------------------------------------------------------------------------------------

#Funções de recompensa
def reward(proximo_estado, estado, acao):
    recompensa = 0

    def verificar_acao_valida(estado,acao):
        recompensa = 0

        if estado[acao] != 0:
            recompensa = recompensa -1000
        else:
            recompensa = recompensa + 25

        return recompensa
        
    def derrota_imediata(estado,acao):
        global todas_combinacoes
        recompensa = 0
        
        for i in todas_combinacoes:
            if ((estado[i[0]] == 2 and estado[i[1]]) == 2) and acao != i[2]:
                print("Derrota")
                recompensa = recompensa - 20
                break

            else:
                continue

        return recompensa


    def vitoria_imediato(proximo_estado):
        recompensa = 0
        
        if done(proximo_estado) == (True, 1):
            recompensa = recompensa + 10

        else:
            pass

        return recompensa

    recompensa = vitoria_imediato(proximo_estado) + verificar_acao_valida(estado,acao) + derrota_imediata(estado,acao)

    return recompensa


def epsilon_greedy_policy(estado, Q):
    global EPSILON

    if random.uniform(0,1) < EPSILON:
        acao = random.choice(range(len(ACOES))) 
        return acao
    #Iremos bloquear jogar nos quadrados também aqui, se não conseguir usar ação aleatória
    else:
        acao = max(ACOES, key = lambda x: Q[(tuple(estado),x)])
        return acao


def play_jogador_2(estado):
    indices_zeros = []
    indices_zeros = [indice for indice, valor in enumerate(estado) if valor == 0]

    if len(indices_zeros) == 0:
        pass
    else:
        zero_aleatorio = random.choice(indices_zeros)

        estado[zero_aleatorio] = 2
        return estado


num_passos = 100

def gerar_episodio(Q):
    episodio = []

    estado = [0,0,0,0,0,0,0,0,0]
    

    for t in range(num_passos):
        print(estado)

        acao = epsilon_greedy_policy(estado, Q)

        #perform the selected action and store the next state information
        proximo_estado = novo_estado(estado,acao)

        recompensa_v = reward(proximo_estado, estado,acao)
        is_done = done(proximo_estado)
        velha = deu_velha(proximo_estado)

        if estado[acao] != 0:
            recompensa_v = -10000
            episodio.append((estado, acao, recompensa_v))
            break
        else:
            #store the state, action, reward in the episode list
            episodio.append((estado, acao, recompensa_v))

            if is_done is not False or velha is True:
                break

            else:            
                estado = proximo_estado

                estado = play_jogador_2(estado)

    return episodio


num_iterations = 100000

#for each iteration
for i in range(num_iterations):
    retorno_total = {}
    
    #so, here we pass our initialized Q function to generate an episode
    episodio = gerar_episodio(Q)
    
    #get all the state-action pairs in the episode
    todo_par_estado_acao = [(tuple(s), a) for (s,a,r) in episodio]
    
    #store all the rewards obtained in the episode in the rewards list
    recompensas = [r for (s,a,r) in episodio]

    #for each step in the episode 
    for t, (estado, acao, recompensa) in enumerate(episodio):
            
        #compute the return R of the state-action pair as the sum of rewards
        R = sum(recompensas[t:])

        #if the state-action pair is occurring for the first time in the episode
        if not (tuple(estado), acao) in todo_par_estado_acao[0:t]:
            #update total return of the state-action pair
            retorno_total[(tuple(estado),acao)] = R 

        else:   
            #update total return of the state-action pair
            retorno_total[(tuple(estado),acao)] = retorno_total[(tuple(estado),acao)] + R 
        
        #update the number of times the state-action pair is visited
        N[(tuple(estado), acao)] += 1

        #compute the Q value by just taking the average
        Q[(tuple(estado),acao)] = retorno_total[(tuple(estado), acao)] / N[(tuple(estado), acao)]

df = pd.DataFrame(Q.items(),columns=['state_action pair','value'])
df = df.sort_values(by='value',ascending=False)
#df.head(10)
for i in range(0,9):
    print(df[df['state_action pair'] == ((1, 0, 2, 1, 0,1,2,0,0), i)])

# Crie um dicionário para rastrear a melhor ação para cada tupla de estado
melhor_acao_por_tupla = {}

# Itere sobre os pares estado-ação no dicionário Q
for (estado, acao), valor in Q.items():
    if estado not in melhor_acao_por_tupla:
        melhor_acao_por_tupla[estado] = (acao, valor)
    elif valor > melhor_acao_por_tupla[estado][1]:
        melhor_acao_por_tupla[estado] = (acao, valor)

# Agora, melhor_acao_por_tupla conterá a melhor ação (e seu valor) para cada tupla de estado
#print(melhor_acao_por_tupla[(0, 0, 0, 0, 1, 0,0,0, 2)])


def acao_ia(estado_ingame):
    acao = (melhor_acao_por_tupla[tuple(estado_ingame)])[0]

    if estado_ingame[acao] == 0:
        return acao
    else:
        acoes_possiveis = [acao for acao, _ in Q.items() if acao[0] == estado]

        # Ordene as ações possíveis pelo valor Q em ordem decrescente
        acoes_ordenadas = sorted(acoes_possiveis, key=lambda acao: Q[acao], reverse=True)

        for i in range(1,9):
            acao = (acoes_ordenadas[i])[1]

            if estado_ingame[acao] == 0:
                return acao
            
            else: 
                continue

estado_ingame = [0,0,0,0,0,0,0,0,0]

def mostrar_estado(estado):
    print(estado[0],estado[1],estado[2])
    print(estado[3],estado[4],estado[5])
    print(estado[6],estado[7],estado[8],"\n")

for i in range(0,9):
    ia_play = acao_ia(estado_ingame)
    estado_ingame = novo_estado(estado_ingame,ia_play)
    mostrar_estado(estado_ingame)
    if done(estado_ingame) is False:
        p1 = int(input("Faça sua jogada: "))
        estado_ingame[(p1-1)] = 2
        mostrar_estado(estado_ingame)
        if done(estado_ingame) is False:
            continue
    else:
        break
