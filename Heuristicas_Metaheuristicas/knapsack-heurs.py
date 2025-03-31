import os
import sys
import numpy as np
from mip import Model, xsum,  maximize, CBC, OptimizationStatus, BINARY
from time import time
#np.random.seed(0)
#np.random.seed(1000)
#np.random.seed(5558)
np.random.seed(15958)
#np.random.seed(20000)

def main():
    assert len(sys.argv) > 1, 'please,provide a data file'
    inst = CInstance(sys.argv[1])

    #inst.print()

    mod = CModel(inst)
    opt = mod.run()

    print('Blank solution')
    sol1 = CSolution(inst)
    sol2 = CSolution(inst)
    solgreedy = CSolution(inst)

    constr = CConstructor()

    print()    
    print('random 1')
    constr.random_solution(sol1)
    sol1.print()

    print()    
    #print('random 2')
    constr.random_solution2(sol2)
    #sol2.print()

    #print()    
    #print('greedy ')
    constr.greedy(solgreedy)
    #solgreedy.print()


    ls = CBuscaLocal()
 
    print('improvement phase')   
    print('random 1')
    ls.swap_one_bit_best_improvement(sol1)
    sol1.print()
    print(f'optimality distance : {100.0*(1-sol1.obj/opt):10.2f} %')
    ls.swap_two_bits_best_improvement(sol1)
    sol1.print()
    print(f'optimality distance : {100.0*(1-sol1.obj/opt):10.2f} %')
    ls.swap_inout_best_improvement(sol1)
    sol1.print()
    print(f'optimality distance : {100.0*(1-sol1.obj/opt):10.2f} %')

    
    print()
    print('random 2')
    ls.swap_one_bit_best_improvement(sol2)
    sol2.print()
    print(f'optimality distance : {100.0*(1-sol2.obj/opt):10.2f} %')
    ls.swap_two_bits_best_improvement(sol2)
    sol2.print()
    print(f'optimality distance : {100.0*(1-sol2.obj/opt):10.2f} %')
    ls.swap_inout_best_improvement(sol2)
    sol2.print()
    print(f'optimality distance : {100.0*(1-sol2.obj/opt):10.2f} %')

    print()
    print('greedy ')
    ls.swap_one_bit_best_improvement(solgreedy)
    solgreedy.print()
    print(f'optimality distance : {100.0*(1-solgreedy.obj/opt):10.2f} %')
    ls.swap_two_bits_best_improvement(solgreedy)
    solgreedy.print()
    print(f'optimality distance : {100.0*(1-solgreedy.obj/opt):10.2f} %')
    ls.swap_inout_best_improvement(solgreedy)
    solgreedy.print()
    print(f'optimality distance : {100.0*(1-solgreedy.obj/opt):10.2f} %')


    sys.exit()


    '''
    sol2 = CSolution(inst)
    solg = CSolution(inst)

    
    # aleatoria
    print('construction phase')   
    print('random 1')
    constr.random_solution(sol)
    sol.print()


    

    


    print('greedy')
    ls.swap_two_bits_best_improvement(solg)
    solg.print()
    
    print('improvement phase')   
    print('random')
    ls.swap_one_bit_best_improvement(sol)
    sol.print()

    print('random 2')
    ls.swap_one_bit_best_improvement(sol2)
    sol2.print()


    #sys.exit()
    '''
  
    ''' 
    print('improvement phase 2')   
    print('random')
    ls.swap_two_bits_best_improvement(sol)
    sol.print()
    

    print('random 2')
    ls.swap_two_bits_best_improvement(sol2)
    sol2.print()

    print('greedy')
    ls.swap_two_bits_best_improvement(solg)
    solg.print()
    ''' 



class CModel():
    def __init__(self,inst):
        self.inst = inst
        self.create_model()

    def create_model(self):
        inst = self.inst
        N = range(inst.n)
        model = Model('Problema da Mochila',solver_name=CBC)
        # variavel: se o projeto j e incluido na mochila
        x = [model.add_var(var_type=BINARY) for j in N]
        # funcao objetivo: maximizar o retorno
        model.objective = maximize(xsum(inst.p[j] * x[j] for j in N))

        # restricao: a capacidade da mochila deve ser respeitada
        model += xsum(inst.w[j] * x[j] for j in N) <= inst.b
        # desliga a impressao do solver
        model.verbose = 0
        self.x = x
        self.model = model

    def run(self):
        inst = self.inst
        N = range(inst.n)
        model,x = self.model,self.x
        # otimiza o modelo chamando o resolvedor
        start = time()
        status = model.optimize()
        end = time()
        # impressao do resultado
        if status == OptimizationStatus.OPTIMAL:
           print("Optimal solution: {:10.2f}".format(model.objective_value))
           print(f'running time    : {end-start:10.2f} s')
           newln = 0
           for j in N:
               if x[j].x > 1e-6:
                   print("{:3d} ".format(j),end='')
                   newln += 1
                   if newln % 10 == 0:
                      newln = 1
                      print()
           print('\n\n')
           return model.objective_value
        return float(-inf)

class CBuscaLocal():
    def __init__(self):
        pass

    def swap_one_bit_best_improvement(self,sol):
        inst = sol.inst
        p,w,M = inst.p,inst.w,inst.M
        b,_b = inst.b,sol._b
        N = np.arange(inst.n)

        best_delta = float('inf')
        best_j = -1

        counter = 0
        while best_delta > 0:
              counter += 1
              best_delta = -float('inf')
              #np.random.shuffle(N)

              for j in N:
                  oldval,newval = sol.x[j], 0 if sol.x[j] else 1
                  delta = p[j] * (newval - oldval)\
                        + M * max(0,_b - b)\
                        - M * max(0,_b + w[j] * (newval - oldval) - b)
                  if delta > best_delta:
                      best_delta = delta
                      best_j = j

              if best_delta > 0:
                  oldval,newval = sol.x[best_j], 0 if sol.x[best_j] else 1
                  sol.x[best_j] = newval 
                  sol.obj += best_delta
                  _b += w[best_j] * (newval - oldval)
                  sol._b = _b
        print(f'counter one bit loop: {counter:3d}')

    def swap_two_bits_best_improvement(self,sol):
        inst = sol.inst
        p,w,M = inst.p,inst.w,inst.M
        b,_b = inst.b,sol._b
        n = inst.n
        N = np.arange(n)

        best_delta = float('inf')
        best_j = -1
        while best_delta > 0:

              best_delta = -float('inf')
              #np.random.shuffle(N)

              h1 = 0
              while h1 < n - 1:
                  j1 = N[h1]
                  oldval1,newval1 = sol.x[j1], 0 if sol.x[j1] else 1

                  h2 = h1 + 1
                  while h2 < n:
                      j2 = N[h2]
                      oldval2,newval2 = sol.x[j2], 0 if sol.x[j2] else 1

                      delta = p[j1] * (newval1 - oldval1)\
                            + p[j2] * (newval2 - oldval2)\
                            + M * max(0,_b - b)\
                            - M * max(0,_b + w[j1] * (newval1 - oldval1) + w[j2] * (newval2 - oldval2) - b)

                      if delta > best_delta:
                         best_delta = delta
                         best_j1 = j1
                         best_j2 = j2

                      h2 += 1
                  h1 += 1

              if best_delta > 0:
                  oldval1,newval1 = sol.x[best_j1], 0 if sol.x[best_j1] else 1
                  oldval2,newval2 = sol.x[best_j2], 0 if sol.x[best_j2] else 1
                  sol.x[best_j1] = newval1 
                  sol.x[best_j2] = newval2 
                  sol.obj += best_delta
                  _b += w[best_j1] * (newval1 - oldval1)\
                      + w[best_j2] * (newval2 - oldval2)
                  sol._b = _b

    def swap_inout_best_improvement(self,sol):
        inst = sol.inst
        n = inst.n
        N = np.arange(n)
        p,w,M = inst.p,inst.w,inst.M
        b,_b,x = inst.b,sol._b,sol.x

        best_delta = float('inf')
        counter = 0
        while best_delta > 0:
              z = np.array([j for j in N if x[j] == 1])
              _z = np.array([j for j in N if x[j] == 0])
              best_delta = -float('inf')
              counter += 1

              for jout in z:
                  for jin in _z:
                      delta = -p[jout] + p[jin]\
                            + M * max(0,_b - b)\
                            - M * max(0,_b + w[jin] - w[jout] - b)
              
                      if delta > best_delta:
                         best_delta = delta
                         best_jout = jout
                         best_jin = jin
                     
              if best_delta > 0:
                 sol.x[best_jin] = 1
                 sol.x[best_jout] = 0
                 sol.obj += best_delta
                 _b += w[best_jin] - w[best_jout]
                 sol._b = _b

class CConstructor():
    def __init__(self):
        pass

    def random_solution2(self,sol):
        inst = sol.inst
        p = np.random.choice(inst.n,1)[0]
        vals = np.random.choice(inst.n,p,replace=False)
        sol.x[:] = 0
        sol.z[:] = -1
        sol.x[vals] = 1
        sol.z[:p] = vals[:]
        sol._b = inst.w[vals].sum()
        sol.get_obj_val()

    def random_solution(self,sol):
        inst = sol.inst
        N = range(inst.n)
        h = 0
        sol._b = 0
        for j in N:
            val = np.random.choice(2,1)[0]
            sol.x[j] = val
            if val > 0:
               sol._b += inst.w[j]
               sol.z[h] = j
               h += 1
        sol.get_obj_val()

    def greedy(self,sol):
        inst = sol.inst
        sortedp = inst.p.argsort()[::-1]
        cumsum = np.cumsum(inst.w[sortedp])
        ind = sortedp[np.argwhere(cumsum <= inst.b).ravel()]
        sol.x[:] = 0
        sol.x[ind] = 1 
        sol.z[:] = -1
        sol.z[:len(ind)] = ind[:]
        sol._b = np.sum(inst.w[ind])
        sol.get_obj_val()

class CSolution():
    def __init__(self,inst):
        self.inst = inst
        self.create_structure()

    def create_structure(self):
        self.x = np.zeros(self.inst.n)
        self.z = np.full(self.inst.n,-1)
        self.obj = 0.0
        self._b = self.inst.b

    def get_obj_val(self):
        inst = self.inst
        p,w,b,M = inst.p,inst.w,inst.b,inst.M
        self._b = (self.x * w).sum()
        self.obj = (self.x * p).sum() - M * max(0,self._b-b)
        return self.obj

    def copy_solution(self,sol):
        sol.x[:] = self.x[:]
        sol.z[:] = self.z[:]
        sol.obj = self.obj
        sol._b = self._b

    def print(self):
        self.get_obj_val()
        print(f'obj  : {self.obj:16.2f}')
        print(f'_b/b : {self._b:16.0f}/{self.inst.b:16.0f}')
        newln = 0
        for j,val in enumerate(self.x):
            if val > 0.9:
                print(f'{j:3d} ',end='')
                newln += 1
                if newln % 10 == 0:
                   newln = 1
                   print()
        print('\n\n')

class CInstance():
    def __init__(self,filename):
        self.read_file(filename)

    def read_file(self,filename):
        self.filename = filename
        assert os.path.isfile(filename), 'please, provide a valid file'
        with open(filename,'r') as rf:
            lines = rf.readlines()
            lines = [line for line in lines if line.strip()]
            self.n = int(lines[0])
            self.b = int(lines[1])
            p,w = [],[]
            for h in range(2,self.n+2):
                _p,_w = [int(val) for val in lines[h].split()]
                p.append(_p),w.append(_w)
            self.p,self.w = np.array(p),np.array(w)
        self.M = self.p.sum()

    def print(self):
        print(f'{self.n:9}')
        print(f'{self.b:9}')
        for h in range(self.n):
            print(f'{self.p[h]:4d} {self.w[h]:4d}')

if __name__ == '__main__':
    main()



