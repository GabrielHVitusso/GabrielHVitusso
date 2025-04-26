# Prova 2 - Heurísticas e Metaheurísticas - Gabriel Hesse Vitusso
# Esse arquivo é a junção das Meta e heurísticas contidas no quiz2.zip, o objetivo dele é organizar o código de forma que ele seja entregue no formato para o relatório, e contenha as analises requesitadas.

import os
import sys
import numpy as np
import numpy.ma as ma
import math
from math import exp,log
from mip import Model, xsum,  maximize, CBC, OptimizationStatus, BINARY
from time import perf_counter as pc

np.random.seed(10200)

def main():
    assert len(sys.argv) > 1, 'please,provide a data file'
    inst = CInstance(sys.argv[1])
    #inst.print()

    sol = CSolution(inst)

    constr = CConstructor()
    
    constr.random_solution(sol)
    sol.print()
    
    ls = CLocalSearch()
    bsol = CSolution(inst)
    for i in range(1):
        ls.vns(sol, 3)
        if sol.obj > bsol.obj:
            bsol.copy(sol)
    bsol.print()
    
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

# Modificado para somente incluir o necessario para o first improvement dos métodos requisitados
class CLocalSearch():
    def __init__(self):
        pass
    def grasp(self, sol, alpha=.10, graspmax=3):
        best_sol = CSolution(sol.inst)
        best_sol.copy(sol)
        constr = CConstructor()
        ls = CLocalSearch()
        h = 0
        while h < graspmax:
            np.random.shuffle(sol.x)
            h += 1
            constr.partial_greedy(sol,alpha)
            #ls.swap_one_bit(sol)
            ls.vnd(sol)#swap_one_bit(sol)
            if sol.obj > best_sol.obj:
                best_sol.copy(sol)
                break
        #best_sol.print()
    
    def vnd(self,sol):
        solstar = CSolution(sol.inst)
        solstar.copy_solution(sol)

        h = 1
        while (h <= 2):
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
    
    def swap_one_bit(self,sol):
        inst = sol.inst
        p,w,M = inst.p,inst.w,inst.M
        b,_b = inst.b,sol._b
        N = np.arange(inst.n)

        best_delta = float('inf')
        best_j = -1

        while best_delta > 0:
              best_delta = -float('inf')
              np.random.shuffle(N)

              for j in N:
                  oldval,newval = sol.x[j], 0 if sol.x[j] else 1
                  delta = p[j] * (newval - oldval)\
                        + M * max(0,_b - b)\
                        - M * max(0,_b + w[j] * (newval - oldval) - b)
                  if delta > 0:
                      best_j = j
                      best_delta = delta
                      oldval,newval = sol.x[best_j], 0 if sol.x[best_j] else 1
                      sol.x[best_j] = newval 
                      sol.obj += best_delta
                      _b += w[best_j] * (newval - oldval)
                      sol._b = _b
                      break

    def swap_two_bits(self, sol):
        inst = sol.inst
        p, w, M = inst.p, inst.w, inst.M
        b, _b = inst.b, sol._b
        N = np.arange(inst.n)

        np.random.shuffle(N)

        for i in N:
            for j in N:
                if i != j:
                    oldval_i, newval_i = sol.x[i], 0 if sol.x[i] else 1
                    oldval_j, newval_j = sol.x[j], 0 if sol.x[j] else 1

                    delta = (p[i] * (newval_i - oldval_i) + p[j] * (newval_j - oldval_j)
                             + M * max(0, _b - b)
                             - M * max(0, _b + w[i] * (newval_i - oldval_i) + w[j] * (newval_j - oldval_j) - b))

                    if delta > 0:
                        for idx in [i, j]:
                            oldval, newval = sol.x[idx], 0 if sol.x[idx] else 1
                            sol.x[idx] = newval
                            sol.obj += p[idx] * (newval - oldval)
                            _b += w[idx] * (newval - oldval)

                        sol._b = _b
                        return

    def swap_bit(self,sol,j):
        inst = sol.inst
        p,w,M,n = inst.p,inst.w,inst.M,inst.n
        b,_b = inst.b,sol._b
        
        oldval,newval = sol.x[j], 0 if sol.x[j] else 1
        delta = p[j] * (newval - oldval)\
              + M * max(0,_b - b)\
              - M * max(0,_b + w[j] * (newval - oldval) - b)
        sol.x[j] = newval 
        sol.obj += delta
        _b += w[j] * (newval - oldval)
        sol._b = _b
        return delta

    def ils(self,sol,\
                max_time,\
                max_iterations,\
                max_perturbation = 5
                ):
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
                 self.vnd(sol)
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

    def sa(self,sol,alpha=0.97,k = 2):
        inst = sol.inst
        p,w,M,n = inst.p,inst.w,inst.M,inst.n
        b,_b = inst.b,sol._b
        N = np.arange(n)
        
        # sa settings
        SAmax = k * n
        initial_temperature = self.sa_initial_temperature(sol,alpha,SAmax)
        temperature = initial_temperature
        final_temperature = 0.01
        n_temp_changes = 0

        # best solution so far
        best_sol = CSolution(inst)
        best_sol.copy(sol)
        #print(f'obj : {sol.obj:10.2f}  b : {sol._b:10.2f}')
        while temperature > final_temperature:
            h = 0
            
            while h < SAmax:
                h += 1
                j = np.random.randint(n)
                delta = self.swap_bit(sol, j)
                np.random.shuffle(N)
            
                if delta > 0:
                    # improving solution
                    # print(f'improving solution obj : {sol.obj:10.2f}  b : {sol._b:10.2f}')
                    # updating to the first solution
                    best_sol.copy(sol)
                    break
                else:
                    rnd = np.random.uniform(0, 1)
                if rnd < exp(delta / temperature):
                    # print(f'worsening solution obj : {sol.obj:10.2f}  b : {sol._b:10.2f}')
                    pass
                else:
                    self.swap_bit(sol, j)
            # diminish temperature
            temperature *= alpha
            # print(f'current temperature:   {temperature:10.2f}')
            n_temp_changes += 1
        #print(f'final temperature               :{temperature:18.2f}')
        #print(f'max number of checked solutions :{n_temp_changes*SAmax:18.0f}') 
        #print(f'existing solutions              :{exp(n*log(2)):18.0f}')

    def sa_initial_temperature(self,sol,alpha,SAmax):
        n = sol.inst.n
        temperature = 2
        #print(f'obj : {sol.obj:10.2f}  b : {sol._b:10.2f}')
        accept = 0
        min_accept = int(alpha*SAmax)
        while accept < min_accept: 
           h = 0
           #print(f'initial temperature:   {temperature:10.2f}')
           while h < SAmax:
                h += 1
                # choose a random position (item) 
                j = np.random.randint(n)
                delta = self.swap_bit(sol,j) 
                if delta > 0:
                   accept += 1
                else: 
                   # if solution is worse than current, accept it if
                   # a given probability is given
                   rnd = np.random.uniform(0,1)
                   if rnd < exp(delta/temperature):
                      accept += 1
                delta = self.swap_bit(sol,j) 
           #print(f"accept {accept:10d} min_accept {min_accept:10d}")
           if accept < min_accept:
              accept = 0
              temperature  *= 1.1
        #print(f'initial temperature:   {temperature:10.2f}')
        return temperature

    def tabu_search(self, sol, tsmax=20):

    
        inst = sol.inst
        p, w, M, n = inst.p, inst.w, inst.M, inst.n
        b, _b = inst.b, sol._b
        N = np.arange(n)
        tssz = 4  # math.ceil(n/3)
        self.tabu_list = np.zeros(n)

        best_sol = CSolution(inst)
        best_sol.copy(sol)

        tsiter = 0
        bestiter = 0
        while (tsiter - bestiter < tsmax):
            tsiter += 1
            np.random.shuffle(N)  # Shuffle the order of bits randomly
            for j in N:
                delta = self.swap_bit(sol, j)
                if (self.tabu_list[j] < tsiter) or (self.tabu_list[j] >= tsiter and sol.obj > best_sol.obj):
                    self.tabu_list[j] = tsiter + tssz
                    if delta > 0:
                        break  # First improvement found
                    else:
                        self.swap_bit(sol, j)  # Revert the change if no improvement
            if sol.obj > best_sol.obj:
                best_sol.copy(sol)
                bestiter = tsiter
        sol.copy(best_sol)

    def vns(self,sol,max_time,strategy='first'):
        crono = Crono()
        solstar = CSolution(sol.inst)
        solstar.copy_solution(sol)
        while crono.get_time() < max_time:
            h = 1
            while h <= 2:
                if strategy == 'best':
                    if h == 1:
                        self.random_swap_one_bit(sol)
                    elif h == 2:
                        self.random_swap_two_bits(sol)
                    else:
                        break
                elif strategy == 'first':
                    if h == 1:
                        self.swap_one_bit(sol)  # First Improvement para 1 bit
                    elif h == 2:
                        self.swap_two_bits(sol)  # First Improvement para 2 bits
                    else:
                        break
                self.vnd(sol)

                if sol.obj > solstar.obj:
                   solstar.copy_solution(sol)
                   h = 1
                else:
                   h += 1
        sol.copy_solution(solstar)


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
        
    def partial_greedy(self,sol,alpha):
        inst = sol.inst
        sol.reset()

        N = range(inst.n)

        stop = False
        ls = CLocalSearch()

        rb = np.zeros(inst.n)

        while stop == False:

            for j in N:
                if sol.x[j] == False:
                    delta = ls.swap_bit(sol,j)
                    rb[j] = sol.obj 
                    delta = ls.swap_bit(sol,j)

            masked = ma.masked_array(rb,mask=sol.x)
            maxrb = masked.max()
            minrb = masked.min()
            interval = maxrb - alpha * (maxrb - minrb)

            items = ma.where(masked >= interval)[0]

            if len(items) > 0 and maxrb > 1e-6:
               j = np.random.choice(items,1)[0]
               ls.swap_bit(sol,j)
               if sol.obj < 1e-6:
                  ls.swap_bit(sol,j)
                  stop = True
            else: 
                stop = True

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
    
    def copy(self,sol):
        self.x[:] =  sol.x[:]
        self.z[:] =  sol.z[:]

        self.obj  =  sol.obj 

        self._b   =  sol._b  
    
    def reset(self):
        self.x[:] =  0
        self.z[:] =  -1
        self.obj  =  0.0
        self._b   =  0.0

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

# Criação de funções para 

if __name__ == '__main__':
    main()

