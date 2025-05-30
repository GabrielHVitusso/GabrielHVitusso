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
import pandas as pd


def main():
    tabela = pd.DataFrame(columns=['Instance', 'Seed', 'Multistart', 'ILS', 'GRASP', 'Tabu Search', 'SA', 'VNS'])
    for i in range(0, 11):  # Iterate from 0 to 10
        instance_file = f"./instances/s{i:03d}.kp"  # Format the instance file name
        inst = CInstance(instance_file)
        
        sol = CSolution(inst)
        constr = CConstructor()
    
        mod = CModel(inst)
        mod.run()
        otimo = mod.model.objective_value
    
        # Initialize a DataFrame to store results
        results = pd.DataFrame(columns=['Seed', 'Multistart', 'ILS', 'GRASP', 'Tabu Search', 'SA', 'VNS'])

        it = 1
        matricula = 2022056242
        # Loop through seeds from 100 to 3000, iterating by 100
        for seed in range(100, 3001, 100):
            np.random.seed(matricula+seed)
            constr.random_solution(sol) 
            print(it)
            # Initialize local search object
            ls = CLocalSearch()

            # Run each algorithm and store the results
            #print("multistart")
            crono = Crono()
            crono.start()
            ls.multistart(sol)
            multistart_time = crono.get_time()
            multistart_obj = sol.obj
            if it == 1 or multistart_obj > results['Multistart'].max():
                best_multistart_obj = multistart_obj
                best_multistart_time = multistart_time
                best_multistart_seed = seed

            #print("ils")
            constr.random_solution(sol) 
            crono.start()
            ls.ils(sol,0.5,5,5,strategy='first')
            ils_time = crono.get_time()
            ils_obj = sol.obj
            if it == 1 or ils_obj > results['ILS'].max():
                best_ils_obj = ils_obj
                best_ils_time = ils_time
                best_ils_seed = seed

            #print("grasp")
            constr.random_solution(sol) 
            crono.start()
            ls.grasp(sol, alpha=0.1, graspmax=10)
            grasp_time = crono.get_time()
            grasp_obj = sol.obj
            if it == 1 or grasp_obj > results['GRASP'].max():
                best_grasp_obj = grasp_obj
                best_grasp_time = grasp_time
                best_grasp_seed = seed

            #print("tabu search")
            constr.random_solution(sol) 
            crono.start()
            ls.tabu_search(sol, tsmax=100000000, max_time=1, tssz= 40)
            tabu_time = crono.get_time()
            tabu_obj = sol.obj
            if it == 1 or tabu_obj > results['Tabu Search'].max():
                best_tabu_obj = tabu_obj
                best_tabu_time = tabu_time
                best_tabu_seed = seed

            #print("sa")
            constr.random_solution(sol) 
            crono.start()
            ls.sa(sol,0.95,10)
            sa_time = crono.get_time()
            sa_obj = sol.obj
            if it == 1 or sa_obj > results['SA'].max():
                best_sa_obj = sa_obj
                best_sa_time = sa_time
                best_sa_seed = seed

            #print("vns")
            constr.random_solution(sol) 
            crono.start()
            ls.vns(sol, max_time=1, strategy='first')
            vns_time = crono.get_time()
            vns_obj = sol.obj
            if it == 1 or vns_obj > results['VNS'].max():
                best_vns_obj = vns_obj
                best_vns_time = vns_time
                best_vns_seed = seed

            # Calculate gaps
            gap_multistar = ((otimo - best_multistart_obj) / otimo) * 100
            gap_ils = ((otimo - best_ils_obj) / otimo) * 100
            gap_grasp = ((otimo - best_grasp_obj) / otimo) * 100
            gap_tabu = ((otimo - best_tabu_obj) / otimo) * 100
            gap_sa = ((otimo - best_sa_obj) / otimo) * 100
            gap_vns = ((otimo - best_vns_obj) / otimo) * 100

            # Append the results to the DataFrame
            results = pd.concat([results, pd.DataFrame([{
            'Instance': i,
            'Seed': seed,
            'Multistart': multistart_obj,
            'gap_multistar': gap_multistar,
            'ILS': ils_obj,
            'gap_ils': gap_ils,
            'GRASP': grasp_obj,
            'gap_grasp': gap_grasp,
            'Tabu Search': tabu_obj,
            'gap_tabu': gap_tabu,
            'SA': sa_obj,
            'gap_sa': gap_sa,
            'VNS': vns_obj,
            'gap_vns': gap_vns
            }])], ignore_index=True)
            it += 1
            # Save the best results for each algorithm in a separate table
        best_results = pd.DataFrame([{
            'Instance': i,
            'best_multistart_seed': best_multistart_seed,
            'Best Multistart': best_multistart_obj,
            'Multistart Time': best_multistart_time,
            'Best ILS Seed': best_ils_seed,
            'Best ILS': best_ils_obj,
            'ILS Time': best_ils_time,
            'Best GRASP Seed': best_grasp_seed,
            'Best GRASP': best_grasp_obj,
            'GRASP Time': best_grasp_time,
            'best_tabu_seed': best_tabu_seed,
            'Best Tabu Search': best_tabu_obj,
            'Tabu Search Time': best_tabu_time,
            'Best SA Seed': best_sa_seed,
            'Best SA': best_sa_obj,
            'SA Time': best_sa_time,
            'best_vns_seed': best_vns_seed,
            'Best VNS': best_vns_obj,
            'VNS Time': best_vns_time
        }])

        # Append the best results to a separate CSV file
        best_results.to_csv('best_results.csv', mode='a', header=not os.path.exists('best_results.csv'), index=False)
        tabela = pd.concat([tabela, results], ignore_index=True)
        
        # Save the results to a CSV file
        tabela.to_csv('results.csv', index=False)
    
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
 
    def multistart(self, sol):
        inst = sol.inst
        p, w, M, n = inst.p, inst.w, inst.M, inst.n
        b, _b = inst.b, sol._b
        N = np.arange(n)
        best_sol = CSolution(inst)
        best_sol.copy(sol)
        constr = CConstructor()
        ls = CLocalSearch()
        for i in range(30):
            constr.random_solution2(sol)
            self.vnd(sol, strategy='first')  # Use 'first' improvement strategy
            #print(sol.obj)
            if sol.obj > best_sol.obj:
                best_sol.copy(sol)
        # best_sol.print()
    
    def grasp(self, sol, alpha=.10, graspmax=3):
        best_sol = CSolution(sol.inst)
        best_sol.copy(sol)
        constr = CConstructor()
        ls = CLocalSearch()
        h = 0
        while h < graspmax:
            h += 1
            constr.partial_greedy(sol,alpha)
            #ls.swap_one_bit(sol)
            ls.vnd(sol)#swap_one_bit(sol)
            if sol.obj > best_sol.obj:
                best_sol.copy(sol)
        #best_sol.print()
    
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
                max_perturbation = 5,
                strategy='best'):
        crono = Crono()
        solstar = CSolution(sol.inst)
        solstar.copy_solution(sol)

        h,gh = 0,0
        self.vnd(sol, strategy=strategy)
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

    def tabu_search(self,sol,tsmax=500000,max_time=3, tssz=4):
        inst = sol.inst
        self.tabu_list = np.zeros(inst.n)
        

        best_sol = CSolution(inst)
        best_sol.copy(sol)
                     
        timer = Crono()
        
        tsiter = 0
        while tsiter < tsmax and timer.get_time() < max_time:
            tsiter += 1
            delta,j = self.tabu_search_first(sol,best_sol,tsiter)
          
            if j == -1:
                continue
        
            self.tabu_list[j] = tsiter + tssz
        
            if sol.obj > best_sol.obj:
                best_sol.copy(sol)
        
        sol.copy_solution(best_sol)

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
    
    def tabu_search_first(self,sol,best_sol,tsiter):
        inst = sol.inst
        N = np.arange(inst.n)
        
        np.random.shuffle(N)
        ls = CLocalSearch()
        
        for j in N:
            delta = ls.swap_bit(sol, j)
            if(tsiter >= self.tabu_list[j]) or (tsiter < self.tabu_list[j] and sol.obj > best_sol.obj):
                if delta > 0:
                    return delta,j
            ls.swap_bit(sol, j)
        return 0, -1
        
    def vns(self,sol,max_time,strategy='first'):
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

if __name__ == '__main__':
    main()

