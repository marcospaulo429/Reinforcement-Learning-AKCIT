import pandas as pd
import numpy as np
import random
from collections import defaultdict
from itertools import permutations
import tkinter as tk
import tkinter.messagebox
from tkinter.tix import COLUMN

Q = defaultdict(float)

#Inicializando a política e a value function:
TAMANHO_QUADRADO = 3
ACOES = (0,1,2,3,4,5,6,7,8)

# Parâmetros
EPSILON = 0.4  # Probabilidade de exploração
GAMMA = 0.8    # Fator de desconto
ALPHA = 1

def done(estado):
    if((estado[0]== 1 and estado[1]==1 and estado[2]==1) or
        (estado[3]== 1 and estado[4]==1 and estado[5]==1) or
        (estado[6]==1 and estado[7]==1 and estado[8]==1) or
        (estado[0]==1 and estado[3]==1 and estado[6]==1) or
        (estado[2]==1 and estado[4]==1 and estado[7]==1) or
        (estado[2]==1 and estado[5]==1 and estado[8]==1) or
        (estado[0]==1 and estado[4]==1 and estado[8]==1) or
        (estado[2]==1 and estado[4]==1 and estado[6]==1)):
        return True, 1

    elif((estado[0]==2 and estado[1]==2 and estado[2]==2) or
        (estado[3]==2 and estado[4]==2 and estado[5]==2) or
        (estado[6]==2 and estado[7]==2 and estado[8]==2)  or
        (estado[0]==2 and estado[3]==2 and estado[6]==2) or
        (estado[2]==2 and estado[4]==2 and estado[7]==2) or
        (estado[2]==2 and estado[5]==2 and estado[8]==2) or
        (estado[0]==2 and estado[4]==2 and estado[8]==2) or
        (estado[2]==2 and estado[4]==2 and estado[6]==2)):

        return True, 2

    else:
        return False

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

def reward(proximo_estado, estado, acao):
    recompensa = 0

    if done(proximo_estado) == (True, 1):
            recompensa = recompensa + 15
            return recompensa 

    elif done(proximo_estado) == (True,2):
        recompensa = recompensa - 10
        return recompensa
    
    else:
         return recompensa
    

def deu_velha(estado):
    num_zeros = estado.count(0)
    if num_zeros <= 1:
        return True
    else:
        return False

def nao_jogar_lugar_adversario(estado,acao):
    if estado[acao] == 0:
        return True

    else:
        return False

def epsilon_greedy_policy(estado, Q):
    if random.uniform(0,1) < EPSILON:
        while True:
            acao = random.choice(range(len(ACOES))) 
            if nao_jogar_lugar_adversario(estado,acao) is True:
                    return acao
            else:
                    continue
            
    else:
        print([(acao, Q[(tuple(estado), acao)]) for acao in ACOES])
        acoes_e_valores = [(acao, Q[(tuple(estado), acao)]) for acao in ACOES]

        # Classificamos a lista em ordem decrescente com base nos valores Q.
        acoes_e_valores_ordenados = sorted(acoes_e_valores, key=lambda x: x[1], reverse=True)

        # Agora, pegamos a ação da lista ordenada.
        for i in range(0,9):
            acao = acoes_e_valores_ordenados[i][0]

            if nao_jogar_lugar_adversario(estado,acao) is True:
                 return acao
                
            else:
                 continue

#Fazendo jogadas aletórias pro jogador 2:
def play_jogador_2(estado):
    indices_zeros = []
    indices_zeros = [indice for indice, valor in enumerate(estado) if valor == 0]

    if len(indices_zeros) == 0:
        return estado
    else:
        zero_aleatorio = random.choice(indices_zeros)

        estado[zero_aleatorio] = 2
        return estado

def estado_pos_acao(estado,acao):
    estado2 = estado.copy()
    estado2[acao] = 1
    estado2 = play_jogador_2(estado2)
    return estado2


num_episodios = 500000
num_timesteps = 30


for i in range(num_episodios):
    estado = [0,0,0,0,0,0,0,0,0]
    acao = epsilon_greedy_policy(estado,Q)

    for t in range(num_timesteps):
        #proximo estado já calculado com as jogadas dos dois jogadores
        proximo_estado = estado_pos_acao(estado,acao)
        recompensa = reward(proximo_estado,estado,acao)
        is_done = done(proximo_estado)
        velha = deu_velha(proximo_estado)
        #print(f"{is_done} , {velha} , {estado}, {proximo_estado}")

        #select the action a dash in the next state using the epsilon greedy policy:
        acao_2 = epsilon_greedy_policy(proximo_estado,Q)

        #compute the Q value of the state-action pair
        Q[(tuple(estado),acao)] += ALPHA * (recompensa + GAMMA * Q[(tuple(proximo_estado),acao_2)]-Q[(tuple(estado),acao)])

        #if the next state is a final state then break the loop else update the next state to the current
        #state
        print(f"{is_done} , {velha} , {estado}, {proximo_estado}")
        if (is_done is not False) or (velha is True):
            print("break")
            break

        else:            
            estado = proximo_estado

            acao = acao_2

df = df.sort_values(by="value", ascending=True)
df.head(100)

# Crie um dicionário para rastrear a melhor ação para cada tupla de estado
melhor_acao_por_tupla = {}

# Itere sobre os pares estado-ação no dicionário Q
for (estado, acao), valor in Q.items():
    if estado not in melhor_acao_por_tupla:
        melhor_acao_por_tupla[estado] = (acao, valor)
    elif valor > melhor_acao_por_tupla[estado][1]:
        melhor_acao_por_tupla[estado] = (acao, valor)

# Agora, melhor_acao_por_tupla conterá a melhor ação (e seu valor) para cada tupla de estado
#print(melhor_acao_por_tupla)


estado_ingame = [0,0,0,0,0,0,0,0,0]

def mostrar_estado(estado):
    print(estado[0],estado[1],estado[2])
    print(estado[3],estado[4],estado[5])
    print(estado[6],estado[7],estado[8],"\n")

for i in range(0,9):
    ia_play = (melhor_acao_por_tupla[tuple(estado_ingame)])[0]
    estado_ingame = novo_estado(estado_ingame,ia_play)
    mostrar_estado(estado_ingame)
    if done(estado_ingame) is False:

        while True:
            p1 = input("Faça sua jogada: ")

            # Verifique se a entrada não está vazia
            if p1.strip():
                p1 = int(p1)
                break  
            else:
                mostrar_estado(estado_ingame)

        estado_ingame[(p1 - 1)] = 2
        mostrar_estado(estado_ingame)
        if done(estado_ingame) is False:
            continue
    else:
        break
