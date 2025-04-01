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
    #dt = CData(matricula,ni,nj,p)
    #print(dt.ni,dt.nj,dt.p,dt.c)
    #mod = CModel(dt)
    #mod.run()
    print("2022056, 100, 50, 5")
    Resolucao(2022056, 100, 50, 5)
    print("2022056, 200, 50, 5")
    Resolucao(2022056, 200, 50, 5)
    print("2022056, 300, 100, 7")
    Resolucao(2022056, 300, 100, 7)
    print("2022056, 400, 100, 7")
    Resolucao(2022056, 400, 100, 7)
    
    print("2022056, 500, 100, 5")
    Resolucao(2022056, 500, 100, 5)
    print("2022056, 500, 100, 7")
    Resolucao(2022056, 500, 100, 7)
    print("2022056, 500, 100, 10")
    Resolucao(2022056, 500, 100, 10)
    
    print("2022056, 600, 100, 5")
    Resolucao(2022056, 500, 100, 5)
    print("2022056, 600, 100, 7")
    Resolucao(2022056, 600, 100, 7)
    print("2022056, 600, 100, 10")
    Resolucao(2022056, 500, 100, 5)
    #print("--------------------------------------------------\n")
    #Aleatoria = CSolution().Aleatoria(dt)
    
    #print("--------------------------------------------------\n")
    #Gulosa = CSolution().Gulosa(dt)
 
   # print("--------------------------------------------------\n")

    #Local = CLocal_Search().Primeira_Melhora(Gulosa)

    
    #print("--------------------------------------------------\n")
    
def Resolucao(matricula,ni,nj,p):
    print("--------------------------------------------------\n")
    
    # Cria-se um CData
    crono = Crono()
    dt_1 = CData(matricula, ni, nj, p)
    print("--------------------------------------------------\n")
    # Faz a Solucao Otima
    mod1 = CModel(dt_1)
    mod1.run()
    print("--------------------------------------------------\n")
    # Faz a solucao gulosa
    Gulosa_1 = CSolution().Gulosa(dt_1)
    print("--------------------------------------------------\n")
    # Faz a busca local
    Local_1 = CLocal_Search().Primeira_Melhora(Gulosa_1)
    print("--------------------------------------------------\n")
    # Faz o GAP
    
    GAP = 100*(mod1.model.objective_value - Local_1.total)/mod1.model.objective_value
    print("GAP: ", GAP)
    
    print("--------------------------------------------------\n")
    # Retorna o tempo de execucao
    
    crono.stop()
    print("Tempo execucao           : {:12.4f} s.".format(crono.elapsed))
 
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
        # Importa valores
        self.dt = dt
        self.ni, self.nj = dt.ni, dt.nj
        self.p = dt.p
        self.d = dt.d
        self.xyi = dt.xyi
        self.xyj = dt.xyj
        self.c = dt.c
        
        # Conta o tempo
        crono = Crono()

        # Cria vetor y e matriz x
        self.y = np.zeros(self.nj, dtype=int)  # Y[nj]
        self.x = np.zeros((self.ni, self.nj))  # X[ni,nj]

        # Escolhe facilidades aleatorias para instalar
        location = np.random.choice(self.nj, self.p, replace=False)
        self.y[location] = 1

        # Distribui a demanda dos clientes aleatoriamente entre as facilidades instaladas
        for i in range(self.ni):
            
            disponivel = location
            # Gera aleatoriamente valores que, quando somados, resultam em 1
            proportions = np.random.dirichlet(np.ones(len(disponivel)))
            # Salva esses valores na coordenada correspondente
            for idx, j in enumerate(disponivel):
                self.x[i, j] = proportions[idx]

        crono.stop()    
        self.print_solution(crono)
        
        return self
    
    def Gulosa(self, dt):
        # Importa valores
        self.dt = dt
        self.ni, self.nj = dt.ni, dt.nj
        self.p = dt.p
        self.d = dt.d
        self.xyi = dt.xyi
        self.xyj = dt.xyj
        self.c = dt.c
        
        # Conta o tempo
        crono = Crono()

        # Cria vetor y e matriz x
        self.y = np.zeros(self.nj, dtype=int)  # Y[nj]
        self.x = np.zeros((self.ni, self.nj))  # X[ni,nj]

        # Lista de locais e clientes
        location = list(range(self.nj))
        clientes = list(range(self.ni))

        # Ordena as facilidades pelo custo total de atender todos os clientes
        custos_em_ordem = [sum(self.c[i, j] * self.d[i] for i in clientes) for j in location]
        locais_ordenados = sorted(location, key=lambda j: custos_em_ordem[j])

        # Escolhe os locais com menor custo, tal que nao ultrapassem p
        locais_escolhidos = locais_ordenados[:self.p]
        self.y[locais_escolhidos] = 1

        # Escolhe de onde atender cada cliente
        for i in clientes:
            # Escolhe o local com menor custo para atender o cliente i
            menor_custo = min(locais_escolhidos, key=lambda j: self.c[i, j])
            self.x[i, menor_custo] = 1  # Toda a demanda do cliente `i` vai para a melhor facilidade
        
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
                
class CLocal_Search():
    
    def Primeira_Melhora(self, sol):
        #Importar a solucao
        self.dt = sol.dt
        self.ni, self.nj = sol.ni, sol.nj
        self.p = sol.p
        self.d = sol.d
        self.xyi = sol.xyi
        self.xyj = sol.xyj
        self.c = sol.c
        self.y = sol.y.copy()
        self.x = sol.x.copy()
        
        # Conta o tempo
        crono = Crono()
    
        #Custo total inicial
        total_inicial = 0
        for i in range(self.ni):
            for j in range(self.nj):
                total_inicial += self.x[i,j] * self.d[i] * self.c[i,j]
                
        # Salva o valor de custo total na solucao
        self.total = total_inicial
        
        auxiliar = 1000000000000000000000 # valor muito grande (BIG M)
        
        while auxiliar>= total_inicial: # Enquanto o resultado nao melhora
            
            resultado = self.movimento() # Faz um movimento
            
            if resultado.total < total_inicial: # Checa se ele melhora
                self.total = resultado.total  # Atualiza o custo total atual
                self.x = resultado.x.copy()  # Atualiza a solução
                self.y = resultado.y.copy()  # Atualiza as facilidades instaladas
                print(f"Iteração {i}: Melhoria encontrada. Novo custo: {self.total:.2f}")
                print(" Melhoria encontrada: Custo {0:.2f} -> {1:.2f}".format(total_inicial, resultado.total))
                break
            
            auxiliar = resultado.total
        
        crono.stop()
        print("Tempo execucao           : {:12.4f} s.".format(crono.elapsed))
        return self

    def movimento(self):
        # Encontra os índices onde y é 1 e onde y é 0
        indices_1 = np.where(self.y == 1)[0]
        indices_0 = np.where(self.y == 0)[0]

        # Verifica se há pelo menos um índice em cada grupo
        if len(indices_1) == 0 or len(indices_0) == 0:
            print("Não há bits suficientes para realizar a troca.")
            return self

        # Seleciona aleatoriamente um índice de cada grupo
        idx_1 = np.random.choice(indices_1)
        idx_0 = np.random.choice(indices_0)

        # Realiza a troca
        self.y[idx_1], self.y[idx_0] = self.y[idx_0], self.y[idx_1]

        # Atualiza x para garantir que apenas colunas com y = 1 tenham valores diferentes de 0
        self.x[:, :] = 0  # deixa o x todo em 0
        locais = np.where(self.y == 1)[0]

        # Escolhe o melhor local para cada cliente i
        for i in range(self.ni):
            
            custo_aux = 1000000000000000000000 # valor muito grande (BIG M)
            melhor = -1
            
            for j in locais:
                if custo_aux > self.c[i, j]:
                    custo_aux = self.c[i, j]
                    melhor = j # Seleciona apenas um local com custo melhor do que o atual
            self.x[i, melhor] = 1 # Aqui ficou o melhor custo valido para locais existentes

        # Calcula o novo custo total
        new_total = 0
        for i in range(self.ni):
            for j in range(self.nj):
                new_total += self.x[i, j] * self.d[i] * self.c[i, j]

        # Devolve a nova solução melhorada ou nao
        result = CLocal_Search()
        result.dt = self.dt
        result.ni, result.nj = self.ni, self.nj
        result.p = self.p
        result.d = self.d
        result.xyi = self.xyi
        result.xyj = self.xyj
        result.c = self.c
        result.y = self.y.copy()
        result.x = self.x.copy()
        result.total = new_total
        # Esse movimento engloba de forma estruturada a criacao de duas vizinhancas, uma em X e outra em Y
        return result

        

if __name__ == '__main__':
   main()

