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

if __name__ == '__main__':
   main()

