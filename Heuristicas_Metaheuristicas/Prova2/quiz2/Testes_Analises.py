# Prova 2 - Heurísticas e Metaheurísticas - Gabriel Hesse Vitusso
# Esse arquivo é a inserção de resultados de Relatorio_prova.py para a definição e calculo de Friedman e de Gráficos de convergência

import os
import sys
import numpy as np
import numpy.ma as ma
import math
from math import exp,log
from mip import Model, xsum,  maximize, CBC, OptimizationStatus, BINARY
from time import perf_counter as pc
import pandas as pd
from scipy.stats import friedmanchisquare
from itertools import combinations
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
import perfprof

def main():
    # Import the results.csv file
    results_path = os.path.join(os.path.dirname(__file__), 'results.csv')
    assert os.path.isfile(results_path), "results.csv file not found"
    
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(results_path)
    #print(df.head())
    
    # Filter columns that do not reference gaps, Seed, or instance
    non_gap_columns = [col for col in df.columns if 'gap' not in col.lower() and 'seed' not in col.lower() and 'instance' not in col.lower()]
    #print("Filtered columns:", non_gap_columns)
    #print("Original columns:", df.columns)
    filtered_df = df[non_gap_columns]
    #print("Filtered DataFrame:\n", filtered_df)
    # Perform the Friedman test
    
    # Ensure the data is in the correct format for the test
    data = [filtered_df[col].values for col in filtered_df.columns]
    stat, p_value = friedmanchisquare(*[filtered_df[col] for col in filtered_df.columns])
    
    
    
    # Print the results with increased decimal precision
    print(f"Friedman test statistic: {stat:.10f}")
    print("P-value: {:.3e}".format(p_value))
    
    # Perform Wilcoxon signed-rank test for all pairs of columns
    column_pairs = list(combinations(filtered_df.columns, 2))
    for col1, col2 in column_pairs:
        stat, p_value = wilcoxon(filtered_df[col1], filtered_df[col2])
        print(f"Wilcoxon test between {col1} and {col2}:")
        print(f"Statistic: {stat:.5e}, P-value: {p_value:.3e}")
    
    # Foi apontado que a seed 2022056242+100 na instancia sk003.kp apresenta gaps em todas as heuristicas, assim, ela foi selecionada para o gráfico de convergência
    matricula = 2022056242
    seed = 100
    np.random.seed(matricula+seed)
    instance_file = f"./instances/s003.kp"
    inst = CInstance(instance_file)
    sol = CSolution(inst)
    constr = CConstructor()
    mod = CModel(inst)
    mod.run()
    sol_otima = mod.model.objective_value
    
    # multistart
    sol_multistart = sol
    constr.random_solution(sol_multistart)
    ls_multistart = CLocalSearch()
    lista_multistart = []
    ls_multistart.multistart(sol_multistart, lista_multistart)
    print(lista_multistart[-1])
    

    # grasp
    sol_grasp = sol
    constr.random_solution(sol_grasp)
    ls_grasp = CLocalSearch()
    lista_grasp = []
    ls_grasp.grasp(sol_grasp, alpha=0.30, graspmax=50, lista_Fos=lista_grasp)    
    print(lista_grasp[-1])
    
    # ils
    sol_ils = sol
    constr.random_solution(sol_ils)
    ls_ils = CLocalSearch()
    lista_ils = []
    ls_ils.ils(sol_ils, max_time=3, max_iterations=1000, max_perturbation=5, strategy='first', lista_Fos=lista_ils)
    print(lista_ils[-1])
    
    # sa
    sol_sa = sol
    constr.random_solution(sol_sa)
    ls_sa = CLocalSearch()
    lista_sa = []
    ls_sa.sa(sol_sa, alpha=0.97, k=2, lista_Fos=lista_sa)
    print(lista_sa[-1])
    
    # tabu search
    sol_tabu = sol
    constr.random_solution(sol_tabu)
    ls_tabu = CLocalSearch()
    lista_tabu = []
    ls_tabu.tabu_search(sol_tabu, tsmax=500000, max_time=3, tssz=4, lista_Fos=lista_tabu)
    print(lista_tabu[-1])
    
    # vns
    sol_vns = sol
    constr.random_solution(sol_vns)
    ls_vns = CLocalSearch()
    lista_vns = []
    ls_vns.vns(sol_vns, max_time=3, strategy='first', lista_Fos=lista_vns)
    print(lista_vns[-1])



    # Plotting the convergence graph
    plt.figure(figsize=(10, 6))
    
    # Function to extend the line horizontally after the last recorded value
    def extend_line(x_values, y_values, last_x):
        if x_values:
            plt.plot(x_values + [last_x], y_values + [y_values[-1]], '-')
        
    # Plotting each algorithm's convergence
    x_multistart = [i[1] for i in lista_multistart]
    y_multistart = [i[0] for i in lista_multistart]
    extend_line(x_multistart, y_multistart, 500)  # Extend to x=500
    plt.plot(x_multistart, y_multistart, label='Multistart')
    
    x_grasp = [i[1] for i in lista_grasp]
    y_grasp = [i[0] for i in lista_grasp]
    extend_line(x_grasp, y_grasp, 500)  # Extend to x=500
    plt.plot(x_grasp, y_grasp, label='Grasp')
    
    x_ils = [i[1] for i in lista_ils]
    y_ils = [i[0] for i in lista_ils]
    extend_line(x_ils, y_ils, 500)  # Extend to x=500
    plt.plot(x_ils, y_ils, label='ILS')
    
    x_sa = [i[1] for i in lista_sa]
    y_sa = [i[0] for i in lista_sa]
    extend_line(x_sa, y_sa, 500)  # Extend to x=500
    plt.plot(x_sa, y_sa, label='SA')
    
    x_tabu = [i[1] for i in lista_tabu]
    y_tabu = [i[0] for i in lista_tabu]
    extend_line(x_tabu, y_tabu, 500)  # Extend to x=500
    plt.plot(x_tabu, y_tabu, label='Tabu Search')
    
    x_vns = [i[1] for i in lista_vns]
    y_vns = [i[0] for i in lista_vns]
    extend_line(x_vns, y_vns, 500)  # Extend to x=500
    plt.plot(x_vns, y_vns, label='VNS')
    
    # Add a horizontal line for the optimal solution value
    plt.axhline(y=sol_otima, color='r', linestyle='--', label='Solução Ótima')
    
    plt.xlabel('Iteration')
    plt.ylabel('Objective Value')
    plt.title('Convergence Graph')
    plt.legend()
    plt.grid(True)

    # Set y-axis limits to zoom in on the relevant range of objective values
    min_y = min(min([i[0] for i in lista_multistart]), min([i[0] for i in lista_grasp]), min([i[0] for i in lista_ils]), min([i[0] for i in lista_sa]), min([i[0] for i in lista_tabu]), min([i[0] for i in lista_vns]))
    max_y = max(max([i[0] for i in lista_multistart]), max([i[0] for i in lista_grasp]), max([i[0] for i in lista_ils]), max([i[0] for i in lista_sa]), max([i[0] for i in lista_tabu]), max([i[0] for i in lista_vns]))
    plt.ylim(min_y * 0.9, max_y * 1.1)  # Adjust the 0.9 and 1.1 factors as needed to zoom

    # Save the plot to a file
    plot_path = os.path.join(os.path.dirname(__file__), 'convergence_graph.png')
    plt.savefig(plot_path)
    
    plt.show()


    # Grafico de performance
    # Prepare the data for the performance profile
    data = {
        'Multistart': [i[0] for i in lista_multistart],
        'Grasp': [i[0] for i in lista_grasp],
        'ILS': [i[0] for i in lista_ils],
        'SA': [i[0] for i in lista_sa],
        'Tabu Search': [i[0] for i in lista_tabu],
        'VNS': [i[0] for i in lista_vns]
    }

    palette = ['-r', ':b', '--c', '-.g', '-y', '-m']
    # Generate the performance profile plot
    perfprof.perfprof(data,palette)

    # Save the plot to a file
    perf_plot_path = os.path.join(os.path.dirname(__file__), 'performance_profile.png')
    plt.savefig(perf_plot_path)

    plt.show()

# Resgate de funções de Relatorio_prova.py para o recalcular de soluções usando seeds selecionadas
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
 
    def multistart(self, sol, lista_Fos = []):
        inst = sol.inst
        p, w, M, n = inst.p, inst.w, inst.M, inst.n
        b, _b = inst.b, sol._b
        N = np.arange(n)
        best_sol = CSolution(inst)
        best_sol.copy(sol)
        constr = CConstructor()
        ls = CLocalSearch()
        
        iter = 0
        
        for i in range(100):
            iter += 1
            constr.random_solution2(sol)
            self.vnd(sol, strategy='first')  # Use 'first' improvement strategy
            #print(sol.get_obj_val())
            if sol.get_obj_val() > best_sol.get_obj_val():
                best_sol.copy(sol)
                lista_Fos.append([sol.get_obj_val(),iter])
            #print(lista_Fos)
        # best_sol.print()
        
    
    def grasp(self, sol, alpha=.10, graspmax=3, lista_Fos = []):
        best_sol = CSolution(sol.inst)
        best_sol.copy(sol)
        constr = CConstructor()
        ls = CLocalSearch()
        h = 0
        iter = 0   
        while h < graspmax:
            h += 1
            iter += 1
            constr.partial_greedy(sol,alpha)
            #ls.swap_one_bit(sol)
            ls.vnd(sol, strategy='first')#swap_one_bit(sol)
            #print(sol.get_obj_val())
            if sol.get_obj_val() > best_sol.get_obj_val():
                best_sol.copy(sol)
                #print(best_sol.get_obj_val())
                lista_Fos.append([sol.get_obj_val(),iter])
                
                
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
                strategy='best',
                lista_Fos = []):
        crono = Crono()
        solstar = CSolution(sol.inst)
        solstar.copy_solution(sol)
        iter = 0

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
                 iter += 1
                 self.perturbation(sol,pert_level)
                 self.vnd(sol,strategy=strategy)
                 #self.rvnd(sol)

                 if sol.get_obj_val() > solstar.get_obj_val():
                    solstar.copy_solution(sol)
                    lista_Fos.append([solstar.get_obj_val(),iter])
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

    def sa(self,sol,alpha=0.97,k = 2, lista_Fos = []):
        inst = sol.inst
        p,w,M,n = inst.p,inst.w,inst.M,inst.n
        b,_b = inst.b,sol._b
        N = np.arange(n)
        iter = 0
        
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
            iter += 1
            while h < SAmax:
                h += 1
                j = np.random.randint(n)
                delta = self.swap_bit(sol,j) 
                if delta > 0:
                   #improving solution
                   #print(f'improving solution obj : {sol.obj:10.2f}  b : {sol._b:10.2f}')
                   #improving best solutio
                   if sol.get_obj_val() > best_sol.get_obj_val():
                       best_sol.copy(sol)
                       lista_Fos.append([sol.get_obj_val(),iter])
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

    def tabu_search(self,sol,tsmax=500000,max_time=3, tssz=4, lista_Fos = []):
        inst = sol.inst
        self.tabu_list = np.zeros(inst.n)
        

        best_sol = CSolution(inst)
        best_sol.copy(sol)
                     
        timer = Crono()
        
        iter = 0
        tsiter = 0
        while tsiter < tsmax and timer.get_time() < max_time:
            tsiter += 1
            delta,j = self.tabu_search_first(sol,best_sol,tsiter)
            iter += 1
            if j == -1:
                continue
        
            self.tabu_list[j] = tsiter + tssz
        
            if sol.get_obj_val() > best_sol.get_obj_val():
                best_sol.copy(sol)
                lista_Fos.append([sol.get_obj_val(),iter])
        
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
        
    def vns(self,sol,max_time,strategy='first', lista_Fos = []):
        crono = Crono()
        solstar = CSolution(sol.inst)
        solstar.copy_solution(sol)
        
        ite = 0
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

                if sol.get_obj_val() > solstar.get_obj_val():
                   solstar.copy_solution(sol)
                   lista_Fos.append([solstar.get_obj_val(), ite])
                   h = 1
                else:
                   h += 1
                
                ite += 1  
        sol.copy_solution(solstar)
        
        return ite
        

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


