import os
import sys
import numpy as np
import numpy.ma as ma
import math
from math import exp,log
from mip import Model, xsum,  maximize, CBC, OptimizationStatus, BINARY

np.random.seed(0)

def main():
    assert len(sys.argv) > 1, 'please,provide a data file'
    inst = CInstance(sys.argv[1])
    #inst.print()

    sol = CSolution(inst)

    constr = CConstructor()
    
    # random
    print('Random solution 1') 
    constr.random_solution(sol)
    sol.print()

    # gulosa 
    #print('greedy ')
    #constr.greedy(sol)
    #sol.print()
    
    ls = CLocalSearch()
    #ls.grasp(sol,alpha=.1,graspmax=20)

    #ls.ga(inst,population_size=200,max_generation=200,prob_xor=0.90,prob_mutation=0.2)

    print('tabu search 1')
    for h in range(30):   
        ls.tabu_search(sol,20)
    sol.print()
    
    print('Random solution 2')
    constr.random_solution2(sol)
    sol.print()
    
    ls = CLocalSearch()
    
    print('tabu search 2')
    for h in range(30):
        ls.tabu_search(sol,20)
    sol.print()
    
    
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

class CLocalSearch():
    def __init__(self):
        pass

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

    def ga(self,inst,population_size=10,max_generation=10,prob_xor=0.50,prob_mutation=.10):
        def repair_solution(sol):
           if sol.obj < 0:
              self.swap_one_bit(sol)
           while sol.obj < 0:
               idx = np.asarray(sol.x == 1).nonzero()[0]

               j = np.random.choice(idx,1)[0]

               self.swap_bit(sol,j)



        def roulette(population,population_size,prob):
            totalobj = sum([p.obj for p in population])

            for j,p in enumerate(population):
                prob[j] = p.obj /totalobj

            idx = np.random.choice(len(population),population_size,replace=False,p=prob)

            idx.sort()

            h = 0

            while h < population_size:
                s = idx[h]
                population[h].copy(population[s])
                h += 1


        if population_size % 2 != 0:
           print('population size must be even. Population will be increased')
           population_size += 1
      
        nchildren = int(population_size/2) if int( population_size/2 ) % 2 == 0 else int(population_size/2) + 1

        P = [CSolution(inst) for p in range(population_size + nchildren)]
        prob = np.zeros(len(P)) 

        constr = CConstructor()

        best_sol = CSolution(inst)
        constr.partial_greedy(best_sol,alpha=.10)
        #constr.random_solution(best_sol)

        
        for h,p in enumerate(P[:population_size]):
            constr.partial_greedy(p,alpha=.10)
            #constr.random_solution(p)

            repair_solution(p)

            if p.obj > best_sol.obj:
               best_sol.copy(p)

        best_sol.print()

        ngenerations = 0
        while ngenerations < max_generation:
            ngenerations += 1

            h = 0
            while h < nchildren:
               j1,j2 = np.random.choice(population_size,2,replace=False)

               if np.random.uniform(0,1) < prob_xor:
                  point = np.random.randint(1,inst.n-2)

                  # crossover
                  P[population_size + h].copy(P[j1]) 
                  P[population_size + h + 1].copy(P[j2])

                  P[population_size + h].x[point:] = P[j2].x[point:]
                  P[population_size + h].get_obj_val()

                  P[population_size + h + 1].x[point:] = P[j1].x[point:]
                  P[population_size + h + 1].get_obj_val()

                  # mutation
                  if np.random.uniform(0,1) < prob_mutation:
                     j = np.random.randint(inst.n)

                     self.swap_bit(P[population_size + h],j)

                  if np.random.uniform(0,1) < prob_mutation:
                     j = np.random.randint(inst.n)

                     self.swap_bit(P[population_size + h + 1],j)

                  # repair solution if infeasible 
                  if P[population_size + h].obj < 0:
                     repair_solution(P[population_size + h])

                  if P[population_size + h + 1].obj < 0:
                     repair_solution(P[population_size + h + 1])
              
                  # memetico
                  #self.swap_one_bit(P[population_size + h])
                  #self.swap_one_bit(P[population_size + h + 1])

                  # update best solution
                  if P[population_size + h].obj > best_sol.obj:
                    best_sol.copy(P[population_size + h])

                  if P[population_size + h+1].obj > best_sol.obj:
                    best_sol.copy(P[population_size + h + 1])
 
                  # next children
                  h += 2
            # surviving solutions
            roulette(P,population_size,prob)
 
        best_sol.print()      


    def grasp(self, sol, alpha=.10, graspmax=3):
        best_sol = CSolution(sol.inst)
        best_sol.copy(sol)
        constr = CConstructor()
        ls = CLocalSearch()
        h = 0
        while h < graspmax:
            h += 1
            constr.partial_greedy(sol,alpha)
            ls.swap_one_bit(sol)
            if sol.obj > best_sol.obj:
                best_sol.copy(sol)
        best_sol.print()

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

    def tabu_search_best_neighbor(self,sol,best_sol,tsiter):
        inst = sol.inst
        p,w,M,n = inst.p,inst.w,inst.M,inst.n
        b,_b = inst.b,sol._b
        N = np.arange(n)
        best_delta,best_j = -float('inf'),-1

        for j in N:
            delta = self.swap_bit(sol,j) 
            if (self.tabu_list[j] < tsiter) or (self.tabu_list[j] >= tsiter and sol.obj > best_sol.obj):
               if best_delta < delta:
                  best_delta,best_j = delta,j
            self.swap_bit(sol,j) 
        return best_delta,best_j

        
    def sa_initial_temperature(self,sol,alpha,SAmax):
        n = sol.inst.n
        temperature = 2
        print(f'obj : {sol.obj:10.2f}  b : {sol._b:10.2f}')
        accept = 0
        min_accept = int(0.90*SAmax)
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
        print(f'initial temperature:   {temperature:10.2f}')
        return temperature

    def sa(self,sol):
        inst = sol.inst
        p,w,M,n = inst.p,inst.w,inst.M,inst.n
        b,_b = inst.b,sol._b
        N = np.arange(n)
        
        # sa settings
        alpha = 0.97
        SAmax = n
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
                delta = self.swap_bit(sol,j) 
                if delta > 0:
                   #improving solution
                   #print(f'improving solution obj : {sol.obj:10.2f}  b : {sol._b:10.2f}')
                   #improving best solutio
                   if sol.obj > best_sol.obj:
                       best_sol.copy(sol)
                else:
                   rnd = np.random.uniform(0,1)
                   if rnd < exp(delta/temperature):
                      #print(f'worsening solution obj : {sol.obj:10.2f}  b : {sol._b:10.2f}')
                      pass
                   else:
                      self.swap_bit(sol,j)
            # diminish temperature
            temperature *= alpha
            #print(f'current temperature:   {temperature:10.2f}')
            n_temp_changes += 1
        print(f'final temperature               :{temperature:18.2f}')
        print(f'max number of checked solutions :{n_temp_changes*SAmax:18.0f}') 
        print(f'existing solutions              :{exp(n*log(2)):18.0f}')

    def swap_one_bit(self,sol):
        inst = sol.inst
        p,w,M = inst.p,inst.w,inst.M
        b,_b = inst.b,sol._b
        N = np.arange(inst.n)

        best_delta = float('inf')
        best_j = -1

        while best_delta > 0:
              np.random.shuffle(N)
              best_delta = -float('inf')

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

        '''
        rb = inst.p/inst.w
        maxrb = relative_benefit.max()
        minrb = relative_benefit.min()
        iterval = maxrb - alpha * (maxrb - minrb)

        items = np.arange(inst.n)
        mask = np.full(inst.n,False,dtype=bool)
        h = 0
        while len(items) > 0:

            masked = ma.masked_array(relative_benefit,mask=mask)

            items = ma.where(masked >= iterval)[0]

            if len(items) > 0:
               j = np.random.choice(items,1)[0]

               sol.x[j] = True
               sol.z[h] = j

               h += 1
               mask[j] = True
            else:
               break

            print(sol.get_obj_val())
        '''

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

    def copy(self,sol):
        self.x[:] =  sol.x[:]
        self.z[:] =  sol.z[:]

        self.obj  =  sol.obj 

        self._b   =  sol._b  

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

if __name__ == '__main__':
    main()



