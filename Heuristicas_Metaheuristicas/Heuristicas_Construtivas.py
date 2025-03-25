# 24_03_2025
# Conteúdo de Inteligencia Computacional para Otimização - 2024 - Marcone Souza

# Bibliotecas
import os
import sys
import numpy as np
from mip import Model, xsum, maximize, CBC, OptimizationStatus, BINARY
from time import time

# Tipos de Problemas Considerados:
## Caixeiro Viajante
## Mochila

# Heuristicas Construtivas, aquelas utilizadas para se encontrar uma solução inicial

## Construção Gulosa : Usada em heurísticas clássicas, estima o benefício de cada elemento e coloca somente o elemento "melhor" a cada passo

class CConstrucao():
    def __init__(self):
        pass
    
    def gulosa(self, sol):
        inst = sol.inst # Pega a instancia da solucao incerida
        sortedp = inst.p.argsort()[::-1] # Ordena os elementos em ordem decrescente do beneficio
        cumsum = np.cumsum(inst.w[sortedp]) # Faz a soma dos pesos
        ind = sortedp[np.argwhere(cumsum <= inst.b).ravel()] # Separa somente os elementos que cabem na mochila e possuem o maior beneficio
        sol.x[:] = 0 # o que nao foi escolhido fica 0
        sol.x[ind] = 1 # o que foi escolhido fica 1
        sol.z[:] = -1 # z quando o objeto nao foi usado
        sol.z[:len(ind)] = ind[:] # z quando o objeto foi usado
        sol._b = np.sum(inst.w[ind]) # Peso final usado
        sol.get_obj_val()
### A grande vantagem dessa heuristica se baseia na simplicidade de implementaca, entretanto, as solucoes obtidas sao de baixa qualidade
### É interessante de se observar que a construção desse algoritmo é feita específicamente para o problema da mochila

## Construcao Aleatoria :Definicao aleatoria de uma solucao

    def aleatoria(self, sol):
        inst = sol.inst
        N = range(inst.n)
        h = 0
        sol._b = 0
        for j in N:
            val = np.random.choice(2,1)[0]
            sol.x[j] = val
            if val > 0:
                sol.b +=inst.w[j]
                sol.z[h] = j
                h += 1
        sol.get_obj_val()
### Essa construcao e com certeza uma da mais rapidas, entretando, nao da para garantir algum tipo de qualidade
### É interessante de se observar que a construção desse algoritmo é feita específicamente para o problema da mochila

## Heuristica do vizinho mais proximo
## Heuristica Bellmore e Nemhauser
## Heuristica da Insercao Mais Barata

# Heuristicas de Refinamento

## Método da Descida/Subida
## Método da Primeira Melhora
## Método de Descida/Subida Randomica
## Descida em Vizinhança Varialvel
## Busca Local