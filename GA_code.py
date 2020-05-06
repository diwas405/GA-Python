# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 11:40:44 2020

@author: Diwas
"""
# =======================================================
#                         Import libraries
# ======================================================= 
import pandas as pd
import random as rand
import numpy as np
from numpy import asarray
from numpy import savetxt
from numpy import cumsum
import matplotlib.pyplot as plt

# =======================================================
#                         Read files
# =======================================================

benefit=pd.read_csv('Benefit.csv')  #read file with benefits
cost=pd.read_csv('Cost.csv')        #read file with cost
cost_d=pd.read_csv('Cost_d.csv')   #read file with cost during project implementation
duration=pd.read_csv('Duration.csv')#read file with duration of project
budget=pd.read_csv('Budget.csv')    #read file with annual budget

# =======================================================
#                         Parameters
# =======================================================
S=49                #specify the number of candiate projects/locations (should match the dataset)
W=13                #specify the planning horizon (should match the dataset)
mutation_rate=0.3   #specify the probability of mutation
crossover_rate=0.8  #specify the portion of population to be replaced after crossover
check_n_gen=100     #specify the number of generations to be checked for max fitness
n_parents=int((crossover_rate*S)/2)*2   #number of parents for crossover (always a even number)
penalty=0.8         # specify the penalty (the fitness becomes penalty*fitness if constraint is violated
popu_size=300       # specify the initial population size
# =======================================================
#                         Fitness funtion
# =======================================================

cost_mat=[]
def fitness_func(solution):
   global cost_mat
   fitness=np.sum(np.sum(np.multiply(pd.DataFrame(solution),(benefit-cost-cost_d))))
   duration_invalid=False
   # ---------- penalty for constraint 2: Complete projects withing planning horizon -----
   for i in range (S):
       #print(count_row)
       try:
           r_count = ''.join([str(x) for x in solution.iloc[i]]).rindex('1')
       except:
           r_count=0
       if r_count != 0 and W-int(r_count)<duration.iloc[i,0]:
           duration_invalid =True
           break
   if duration_invalid:
       fitness=penalty*fitness
   # ---------penalty for constraint 1: Cost is within the budget -----------------
   cost_mat=np.multiply(pd.DataFrame(solution),cost)
   cost_mat=cost_mat.transpose()
   #print(cost_mat)
   budget_invalid=False
   for i in range (W):
       if sum(cost_mat.iloc[i])>budget.iloc[0,i]:
           budget_invalid=True
           break
       elif i+1<W:
           budget.iloc[0,i+1]+=budget.iloc[0,i]-sum(cost_mat.iloc[i])
   if budget_invalid:
       fitness=penalty*fitness
   print(fitness)
   if(not duration_invalid or not budget_invalid):
       print("############Found#########")
   return fitness

# =======================================================
#                         crossover funtion
# =======================================================

# --------------- horizontal crossover ----------------------
def hor_crossover(Parent1, Parent2):
    rand_row=np.random.randint(1,S)
    Offspring1=np.concatenate((Parent1[0:rand_row,],Parent2[rand_row:S,]))
    Offspring2=np.concatenate((Parent2[0:rand_row,],Parent1[rand_row:S,]))
    return Offspring1, Offspring2

# --------------- vertical crossover ----------------------
def ver_crossover(Parent1, Parent2):
    rand_col=np.random.randint(1,W)
    Offspring1=np.concatenate((Parent1[:,0:rand_col],Parent2[:,rand_col:W]),axis=1)
    Offspring2=np.concatenate((Parent2[:,0:rand_col],Parent1[:,rand_col:W]),axis=1)
    return Offspring1, Offspring2

# =======================================================
#                         Mutation funtion
# =======================================================

# --------------- horizontal mutation ----------------------
def hor_mutation(m,S):
    rand_mut_r1=rand.randint(1,S-1)
    rand_mut_r2=rand.randint(1,S-1)
    while rand_mut_r1==rand_mut_r2:
        rand_mut_r2=rand.randint(1,S-1)    
    temp=m[rand_mut_r1]
    m[rand_mut_r1]=m[rand_mut_r2]
    m[rand_mut_r2]=temp
    return m
# --------------- vertical mutation ----------------------            
def ver_mutation(m):
    m=hor_mutation(m.transpose(),W)
    return m.transpose()     
      
#-------- Population intialization -----------------------
init_pop=np.random.randint(2,size=(popu_size,S,W))

#------ function for calculating fitness for pupulation at each iteration ---------
def cal_fitness(init_pop):
    fit_pop=[]
    for i in range (S):
        p=pd.DataFrame(init_pop[i])
        #p.reset_index(inplace=True,drop=True)
        fit_pop.append(fitness_func(p))
    fit_pop=np.array(fit_pop)
    return fit_pop
fit_pop = cal_fitness(init_pop)
fit_pop_prob=cumsum(fit_pop/np.sum(fit_pop))
#print("fit_pop")
# ============================================================================
#      Main loop for the GA (selection, crossover, mutation and solution)
# ============================================================================
count_best=0
best_fit=-1
average_fit=[]
average=[]
while (True):
    # --------------- Selection using Roulette wheel ------------------
    rand_roulette=[]
    parent=[]
    parent_ind=[]
    fit_pop_prob=cumsum(fit_pop/np.sum(fit_pop))   # calculating probabilities for roulette wheel
    while len(parent_ind)<n_parents:
        rand_roulette=rand.uniform(0,1)
        for i in range(S):
            if rand_roulette<=fit_pop_prob[i]:
                if i not in parent_ind:
                    parent_ind.append(i)
                    break
    for parent_i in parent_ind:
        parent.append(init_pop[parent_i])
    # -------------random pairs between parents by shuffling their order ------------
    cross_pair=np.array([i for i in range(n_parents)])
    np.random.shuffle(cross_pair)
    # ------------- Crossover operation between the random parent pairs ---------------
    offspring=[]
    for m in range(0,n_parents,2):
        rand_R=rand.uniform(0,1)
        if rand_R>0.5:
            offs1,offs2=hor_crossover(parent[cross_pair[m]],parent[cross_pair[m+1]])
        else:
            offs1,offs2=ver_crossover(parent[cross_pair[m]],parent[cross_pair[m+1]])
        if (offs1==np.zeros((S,W))).all():
            offs1=parent[cross_pair[m]]
        elif(offs2==np.zeros((S,W))).all():
            offs2=parent[cross_pair[m+1]]
        offspring.append(offs1)
        offspring.append(offs2)
    #----------- replacing worst fit individuals by offsprings -----------------------
    n=np.argpartition(fit_pop,n_parents)[0:n_parents]
    for o in range(n_parents):
        init_pop[n[o]]=offspring[o]
    #print (init_pop)
    #--------------------- Mutation operation----------------------------
    for index, m in enumerate(init_pop):
        rand_mut=rand.uniform(0,1)
        if rand_mut>1-mutation_rate:   # Checking condition whether to mutate or not
            rand_mutVH=rand.uniform(0,1)
            if rand_mutVH<0.5:         # Condition for vertical mutation
                number_mut=int(0.1*len(m)*len(m[0]))
                muta_array=np.array([(x,y) for x in range(len(m)) for y in range(len(m[0]))])
                muta_array_ind=np.array([x for x in range(len(muta_array))])
                np.random.shuffle(muta_array_ind)
                for a in range(number_mut):
                    m[muta_array[muta_array_ind[a]]]=1-m[muta_array[muta_array_ind[a]]]
                mutated = m
            elif rand_mutVH<0.5:
                mutated=ver_mutation(m)
            else:
                mutated=hor_mutation(m,S) # condition for horizontal mutation
            init_pop[index]=mutated
    fit_pop=cal_fitness(init_pop)         # reevaluating fitness after crossover and mutation
    average_fit.append(sum(fit_pop)/len(fit_pop))
    average.append(max(fit_pop))
    # ------- getting the fitness for the best individual and checking end condition ---------
    maxfit=max(fit_pop)
    if maxfit>best_fit:
        count_best=0
        best_fit=maxfit
    else:
        count_best+=1
    if count_best==check_n_gen:
        break
# =======================================================
#                         plotting the fitness values
# =======================================================
plt.plot(average, color='green', linewidth=1, label="Individual")
plt.plot(average_fit, color='blue', linewidth=1, label="Population mean")
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.legend()
# =======================================================
#                         Print the solution
# =======================================================
print("The solution is ",init_pop[np.argmax(fit_pop)])
# =======================================================
#                         Save the solution
# =======================================================
plt.savefig("Fitness.jpg")
print(init_pop[np.argmax(fit_pop)])
np.savetxt('Solution.csv', init_pop[np.argmax(fit_pop)], delimiter=',')