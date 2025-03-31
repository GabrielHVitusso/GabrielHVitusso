from mip import Model, xsum,  minimize, CBC, OptimizationStatus, BINARY
from itertools import product
import matplotlib.pyplot as plt
from math import sqrt
from time import perf_counter as pc

import numpy as np
import scipy
import sys

def main():
    assert len(sys.argv) > 4, 'please, provide <matricula> <ni> <nj> <p>'
    matricula,ni,nj,p = [int(val) for val in sys.argv[1:]]
    dt = CData(matricula,ni,nj,p)
    print(dt.ni,dt.nj,dt.p,dt.c)
    mod = CModel(dt)
    mod.run()
    print("--------------------------------------------------\n")
    Aleatoria = CSolution().Aleatoria(dt)
    
    print("--------------------------------------------------\n")
    Gulosa = CSolution().Gulosa(dt)
    
 
 
class CData:
   def __init__(self,matricula,ni,nj,p):
       np.random.seed(matricula)
       self.p,self.ni,self.nj = p,ni,nj
       self.d = np.random.randint(100,size=(ni,))     # demand of node i
       self.xyi = np.random.randint(300,size=(ni,2))   # coord of node customer i
       self.xyj = np.random.randint(300,size=(nj,2))   # coord of node facilitiy j
       self.c = np.ceil(scipy.spatial.distance.cdist(self.xyi,self.xyj)) # unitary transportation cost

class Crono():
    def __init__(self):
        self.start = pc()

    def stop(self):
        self.elapsed = (pc() - self.start)
        return self.elapsed 

    def start(self):
        self.start = pc()

    def reset(self):
        self.start = pc()

class CModel():
   def __init__(self,dt):
       self.dt = dt
       self.create_model()

   def create_model(self):
       dt = self.dt 
       I,J = range(dt.ni),range(dt.nj)
       model = Model('Problema de Localizacao',solver_name=CBC) 
       # variavel: igual a 1 se uma planta e instalada em j; 0, caso contrario
       y = [model.add_var(var_type=BINARY) for j in J]
       # variavel: proporcao da demanda do cliente i atendida pela facilidade j 
       x = {(i,j) : model.add_var(lb=0.0) for (i,j) in product(I,J)}

       # definicao da funcao objetivo
       # funcao objetivo: minimizar o custo total
       model.objective = minimize( xsum(dt.d[i] * dt.c[i][j] * x[i,j] for (i,j) in product(I,J)))

       # restricoes
       model += xsum(y[j] for j in J) == dt.p
       # restricao: a demanda do cliente deve ser totalmente atendida por alguma facilidade j
       # s.t. restricao_demanda{i in I}: sum{j in J} x[i,j] = 1;
       for i in I:
           model += xsum(x[i,j] for j in J) == 1
           
       # restricao: o cliente i so pode ser atendido por j, se j estiver instalado 
       # s.t. restricao_ativacao{i in I,j in J}: x[i,j] <= y[j]; 
       for (i,j) in product(I,J):
            model += x[i,j] <= y[j]
       self.model,self.x,self.y = model,x,y

   def run(self):
       model,x,y = self.model,self.x,self.y
       dt = self.dt 
       I,J = range(dt.ni),range(dt.nj)
       # otimiza o modelo chamando o resolvedor 
       model.verbose = 0
       crono = Crono()
       status = model.optimize()
       crono.stop()

       # imprime solucao
       if status == OptimizationStatus.OPTIMAL:
           print("Custo total              : {:12.2f}.".format(model.objective_value))
           print("Tempo execucao           : {:12.2f} s.".format(crono.elapsed))
                   
           print( "facilidades : demanda : clientes ")
           for j in J:
               if y[j].x > 1e-6:
                  print("{:11d} : {:7.0f} : ".format(j+1,sum([x[i,j].x * dt.d[i] for i in I])),end='')
                  for i in I:
                      if x[i,j].x > 1e-6: 
                         print(" {:d}".format(i+1),end='')
                  print()

class CSolution():
    
    def Aleatoria(self, dt):
        self.dt = dt
        self.ni, self.nj = dt.ni, dt.nj
        self.p = dt.p
        self.d = dt.d
        self.xyi = dt.xyi
        self.xyj = dt.xyj
        self.c = dt.c
        crono = Crono()

        # Inicializa as variáveis de decisão
        self.y = np.zeros(self.nj, dtype=int)  # Facilidades instaladas
        self.x = np.zeros((self.ni, self.nj))  # Proporção da demanda atendida

        # Seleciona aleatoriamente `p` facilidades para serem instaladas
        selected_facilities = np.random.choice(self.nj, self.p, replace=False)
        self.y[selected_facilities] = 1

        # Distribui a demanda dos clientes aleatoriamente entre as facilidades instaladas
        for i in range(self.ni):
            # Facilidades disponíveis para atender o cliente `i`
            available_facilities = selected_facilities
            # Gera proporções aleatórias para distribuir a demanda
            proportions = np.random.dirichlet(np.ones(len(available_facilities)))
            for idx, j in enumerate(available_facilities):
                self.x[i, j] = proportions[idx]

        crono.stop()    
        self.print_solution(crono)
        
        return self
    
    def Gulosa(self, dt):
        self.dt = dt
        self.ni, self.nj = dt.ni, dt.nj
        self.p = dt.p
        self.d = dt.d
        self.xyi = dt.xyi
        self.xyj = dt.xyj
        self.c = dt.c
        crono = Crono()

        # Inicializa as variáveis de decisão
        self.y = np.zeros(self.nj, dtype=int)  # Facilidades instaladas
        self.x = np.zeros((self.ni, self.nj))  # Proporção da demanda atendida

        # Lista de facilidades e clientes
        facilities = list(range(self.nj))
        clients = list(range(self.ni))

        # Ordena as facilidades pelo custo total de atender todos os clientes
        facility_costs = [
            sum(self.c[i, j] * self.d[i] for i in clients) for j in facilities
        ]
        sorted_facilities = sorted(facilities, key=lambda j: facility_costs[j])

        # Seleciona as `p` facilidades com menor custo
        selected_facilities = sorted_facilities[:self.p]
        self.y[selected_facilities] = 1

        # Distribui a demanda dos clientes para as facilidades selecionadas
        for i in clients:
            # Escolhe a facilidade com menor custo para atender o cliente `i`
            best_facility = min(
                selected_facilities, key=lambda j: self.c[i, j]
            )
            self.x[i, best_facility] = 1  # Toda a demanda do cliente `i` vai para a melhor facilidade
        
        crono.stop()    
        self.print_solution(crono)
        
        return self

    def print_solution(self, crono):
        # Facilidades feitas
        print("Facilidades instaladas (y):", self.y)
        
        #Custo total
        total = 0
        for i in range(self.ni):
            for j in range(self.nj):
                total += self.x[i,j] * self.d[i] * self.c[i,j]
        print("Custo: ", total)
        print("Tempo execucao           : {:12.4f} s.".format(crono.elapsed))
        print("facilidades : demanda : clientes")
        for j in range(self.nj):
            if self.y[j] > 0:  # Verifica se a facilidade foi instalada
                demanda_total = sum(self.x[i, j] * self.d[i] for i in range(self.ni))
                print("{:11d} : {:7.0f} : ".format(j + 1, demanda_total), end='')
                for i in range(self.ni):
                    if self.x[i, j] > 1e-6:  # Verifica se o cliente está sendo atendido pela facilidade
                        print(" {:d}".format(i + 1), end='')
                print()
    
    """def Primeira_Melhoria(self, dt, sol):
        
        #Algoritmo de Primeira Melhoria para refinar a solução inicial.
        #:param dt: Dados do problema (CData)
        #:param sol: Solução inicial (CSolution)
        #:return: Solução refinada
        
        self.dt = dt
        self.ni, self.nj = dt.ni, dt.nj
        self.p = dt.p
        self.d = dt.d
        self.c = dt.c

        # Copia a solução inicial
        self.y = sol.y.copy()
        self.x = sol.x.copy()

        # Calcula o custo inicial
        current_cost = sum(
            self.x[i, j] * self.d[i] * self.c[i, j]
            for i in range(self.ni)
            for j in range(self.nj)
        )

        # Itera sobre a vizinhança
        for j in range(self.nj):
            if self.y[j] == 1:  # Facilidades instaladas
                for k in range(self.nj):
                    if self.y[k] == 0:  # Facilidades não instaladas
                        # Troca a instalação de `j` por `k`
                        new_y = self.y.copy()
                        new_y[j] = 0
                        new_y[k] = 1

                        # Recalcula a alocação de clientes
                        new_x = np.zeros((self.ni, self.nj))
                        for i in range(self.ni):
                            # Seleciona a facilidade instalada com menor custo para o cliente `i`
                            installed_facilities = [f for f in range(self.nj) if new_y[f] == 1]
                            best_facility = min(installed_facilities, key=lambda f: self.c[i, f])
                            new_x[i, best_facility] = 1

                        # Calcula o custo da nova solução
                        new_cost = sum(
                            new_x[i, j] * self.d[i] * self.c[i, j]
                            for i in range(self.ni)
                            for j in range(self.nj)
                        )

                        # Verifica se a nova solução é melhor
                        if new_cost < current_cost:
                            print(f"Melhoria encontrada: Custo {current_cost:.2f} -> {new_cost:.2f}")
                            self.y = new_y
                            self.x = new_x
                            return self  # Retorna a nova solução no mesmo formato

        # Retorna a solução original se nenhuma melhoria for encontrada
        print("Nenhuma melhoria encontrada.")
        return self

    def calculate_total_cost(self, x, y, d, c):
        
        #Calcula o custo total da solução.
        #:param x: Matriz de alocação de clientes
        #:param y: Vetor de facilidades instaladas
        #:param d: Demanda dos clientes
        #:param c: Custos de transporte
        #:return: Custo total
        
        total_cost = 0
        for i in range(len(d)):
            for j in range(len(y)):
                total_cost += x[i, j] * d[i] * c[i, j]
        return total_cost

    def reallocate_clients(self, ni, nj, y, d, c):
        
        #Realoca os clientes para as facilidades instaladas.
        #:param ni: Número de clientes
        #:param nj: Número de facilidades
        #:param y: Vetor de facilidades instaladas
        #:param d: Demanda dos clientes
        #:param c: Custos de transporte
        ##:return: Nova matriz de alocação de clientes
        
        x = np.zeros((ni, nj))
        for i in range(ni):
            # Seleciona a facilidade instalada com menor custo para o cliente `i`
            installed_facilities = [j for j in range(nj) if y[j] == 1]
            best_facility = min(installed_facilities, key=lambda j: c[i, j])
            x[i, best_facility] = 1  # Toda a demanda do cliente `i` vai para a melhor facilidade
        return x"""

if __name__ == '__main__':
   main()

