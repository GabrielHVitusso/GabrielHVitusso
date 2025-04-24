import os
import sys
import numpy as np
from mip import Model, xsum,  maximize, CBC, OptimizationStatus, BINARY
from time import perf_counter as pc

np.random.seed(5000)

def main():
    assert len(sys.argv) > 1, 'please,provide a data file'
    inst = CInstance(sys.argv[1])
    #inst.print()

    sol = CSolution(inst)

    constr = CConstructor()
    
    # aleatoria
    print('construction phase')   
    #constr.random_solution(sol)
    #sol.print()

    #print('random 2')
    #constr.random_solution2(sol)
    #sol.print()

    # gulosa 
    print('greedy ')
    constr.greedy(sol)
    sol.print()
    
    ls = CBuscaLocal()
    print('vnd')   
    ls.vnd(sol)
    sol.print()

    print('vns')   
    solvns = CSolution(inst)
    constr.greedy(solvns)
    solvns.print()
    ls.vns(solvns,3)
    solvns.print()


    print('ils')   
    #ls.ils(sol,5,5,15,strategy='best')
    ls.ils(sol,5,5,5,strategy='first')
    sol.print()
    '''
    print('vns')   
    solvns = CSolution(inst)
    constr.greedy(solvns)
    solvns.print()
    ls.vns(solvns,3)
    solvns.print()
    '''

    mod = CModel(inst)
    mod.run()
     

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
        status = model.optimize()

        # impressao do resultado
        if status == OptimizationStatus.OPTIMAL:
           print("Optimal solution: {:10.2f}".format(model.objective_value))
           newln = 0
           for j in N:
               if x[j].x > 1e-6:
                   print("{:3d} ".format(j),end='')
                   newln += 1
                   if newln % 10 == 0:
                      newln = 1
                      print()
           print('\n\n')

class CBuscaLocal():
    def __init__(self):
        pass

    def rvnd(self,sol):
        solstar = CSolution(sol.inst)
        solstar.copy_solution(sol)
        randn = np.arange(2)
        np.random.shuffle(randn)
        h = 1
        while (h <= 2):
            n = randn[h-1]
            if n == 1:
                self.swap_one_bit_best_improvement(sol)
            elif n == 2:
                self.swap_two_bits_best_improvement(sol)
            else:
               break
            if sol.obj > solstar.obj:
               solstar.copy_solution(sol)
               h = 1
            else:
               h += 1

    def vnd(self,sol,strategy='best'):
        solstar = CSolution(sol.inst)
        solstar.copy_solution(sol)

        h = 1
        while (h <= 2):
            if strategy == 'best':
               if h == 1:
                   self.swap_one_bit_best_improvement(sol)
               elif h == 2:
                   self.swap_two_bits_best_improvement(sol)
               else:
                    break
            else:
               if h == 1:
                  self.swap_one_bit(sol)
               elif h == 2:
                  self.swap_two_bits(sol)
               else:
                  break

            if sol.obj > solstar.obj:
               solstar.copy_solution(sol)
               h = 1
            else:
               h += 1

    
    def vns(self,sol,max_time,strategy='best'):
        crono = Crono()
        solstar = CSolution(sol.inst)
        solstar.copy_solution(sol)
        while crono.get_time() < max_time:
            h = 1
            while h <= 2:
                if h == 1:
                    self.random_swap_one_bit(sol)
                elif h == 2:
                    self.random_swap_two_bits(sol)
                else:
                    break
                self.vnd(sol,strategy=strategy)

                if sol.obj > solstar.obj:
                   solstar.copy_solution(sol)
                   h = 1
                else:
                   h += 1
        sol.copy_solution(solstar)

    def ils(self,sol,\
                max_time,\
                max_iterations,\
                max_perturbation = 5,
                strategy='best'):
        crono = Crono()
        solstar = CSolution(sol.inst)
        solstar.copy_solution(sol)

        h,gh = 0,0
        self.vnd(sol)
        #self.rvnd(sol)
        pert_level = 1
        while h < max_iterations and crono.get_time() < max_time:
          
            k = 0
            while k < max_perturbation:
                 '''
                 print(f'gh {gh:3d}\
 h {h:3d}\
 level {pert_level:3d}\
 liter {k:3d}\
 solstar {solstar.obj:12.2f}\
 sol {sol.obj:12.2f}\
 {crono.get_time():10.2f}s')
                 '''
                 self.perturbation(sol,pert_level)
                 self.vnd(sol,strategy=strategy)
                 #self.rvnd(sol)

                 if sol.obj > solstar.obj:
                    solstar.copy_solution(sol)
                    h = 1
                    k = 1
                    pert_level = 1
                 else:
                    h += 1
                 k += 1
                 gh += 1 
            pert_level += 1
            if pert_level < max_perturbation:
               h = 1
        sol.copy_solution(solstar)
    def perturbation(self,sol,pert_level):
        h = 0
        while h <= pert_level:
            self.random_swap_one_bit(sol)
            h += 1

    def random_swap_one_bit(self,sol):
        inst = sol.inst
        n,p,w,b,_b,M = inst.n,inst.p,inst.w,inst.b,sol._b,inst.M
        idx = np.random.randint(n)
        oldval,newval = sol.x[idx], 0 if sol.x[idx] else 1
        delta = p[idx] * (newval - oldval)\
              + M * max(0,_b - b)\
              - M * max(0,_b + w[idx] * (newval - oldval) - b)
        sol.x[idx] = newval 
        sol.obj += delta
        _b += w[idx] * (newval - oldval)
        sol._b = _b
        
    def random_swap_two_bits(self,sol):
        inst = sol.inst
        n,p,w,b,_b,M = inst.n,inst.p,inst.w,inst.b,sol._b,inst.M
        idx1,idx2 = np.random.choice(n,size=2,replace=False)
        oldval1,newval1 = sol.x[idx1], 0 if sol.x[idx1] else 1
        oldval2,newval2 = sol.x[idx2], 0 if sol.x[idx2] else 1
        delta = p[idx1] * (newval1 - oldval1)\
              + p[idx2] * (newval2 - oldval2)\
              + M * max(0,_b - b)\
              - M * max(0,_b + w[idx1] * (newval1 - oldval1) \
              + w[idx2] * (newval2 - oldval2) - b)
        sol.x[idx1] = newval1 
        sol.x[idx2] = newval2 
        sol.obj += delta
        _b += w[idx1] * (newval1 - oldval1)\
            + w[idx2] * (newval2 - oldval2)
        sol._b = _b
    
    def swap_one_bit(self,sol):
        inst = sol.inst
        p,w,M = inst.p,inst.w,inst.M
        b,_b = inst.b,sol._b
        N = np.arange(inst.n)

        np.random.shuffle(N)
        delta = float('inf')
        while delta > 0:
              delta = -float('inf')

              for j in N:
                  oldval,newval = sol.x[j], 0 if sol.x[j] else 1
                  delta = p[j] * (newval - oldval)\
                        + M * max(0,_b - b)\
                        - M * max(0,_b + w[j] * (newval - oldval) - b)
                  if delta > 0:
                     oldval,newval = sol.x[j], 0 if sol.x[j] else 1
                     sol.x[j] = newval 
                     sol.obj += delta
                     _b += w[j] * (newval - oldval)
                     sol._b = _b
   
    def swap_two_bits(self,sol):
        inst = sol.inst
        p,w,M = inst.p,inst.w,inst.M
        b,_b = inst.b,sol._b
        n = inst.n
        N = np.arange(n)
        np.random.shuffle(N)

        delta = float('inf')

        while delta > 0:

              delta = -float('inf')

              h1 = 0
              while h1 < n - 1:
                  j1 = N[h1]

                  h2 = h1 + 1
                  while h2 < n:
                      j2 = N[h2]
                      oldval1,newval1 = sol.x[j1], 0 if sol.x[j1] else 1
                      oldval2,newval2 = sol.x[j2], 0 if sol.x[j2] else 1

                      delta = p[j1] * (newval1 - oldval1)\
                            + p[j2] * (newval2 - oldval2)\
                            + M * max(0,_b - b)\
                            - M * max(0,_b + w[j1] * (newval1 - oldval1) + w[j2] * (newval2 - oldval2) - b)

                      if delta > 0:
                         oldval1,newval1 = sol.x[j1], 0 if sol.x[j1] else 1
                         oldval2,newval2 = sol.x[j2], 0 if sol.x[j2] else 1
                         sol.x[j1] = newval1 
                         sol.x[j2] = newval2 
                         sol.obj += delta
                         _b += w[j1] * (newval1 - oldval1)\
                             + w[j2] * (newval2 - oldval2)
                         sol._b = _b
                      h2 += 1
                  h1 += 1

    def swap_one_bit_best_improvement(self,sol):
        inst = sol.inst
        p,w,M = inst.p,inst.w,inst.M
        b,_b = inst.b,sol._b
        N = np.arange(inst.n)

        best_delta = float('inf')
        best_j = -1

        while best_delta > 0:
              best_delta = -float('inf')

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
        self.x[:] = sol.x[:]
        self.z[:] = sol.z[:]
        self.obj = sol.obj
        self._b = sol._b

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

class Crono():
    def __init__(self):
        self.start_time = pc()

    def start(self):
        self.reset()

    def get_time(self):
        return (pc() - self.start_time)

    def reset(self):
        self.start_time = pc()

if __name__ == '__main__':
    main()



