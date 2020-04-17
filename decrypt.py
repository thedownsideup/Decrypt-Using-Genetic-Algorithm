#!/usr/bin/env python
# coding: utf-8

# ### Artificial Intelligence Project #2 : Decrypt Using Genetic Algorithm
# Mahsa Eskandari Ghadi

# In this project we use genetic algorithm find the best solution amongst many. What we are looking for (AKA the solution) is the key that determines which letter is mapped to which letter of the alphabet, this is called Substitution Cipher. The reason we use genetic algorithm for this is that they are primarily used to efficiently search a large problem space similar to what we have here. Further introductory explanations can be found in the project description. We also answer some questions written in green like below. <br>
# 
# <font color="29AB87"><b>How is a clean data achieved in a project like this? </b></font> Cleane data here can be achieved by for example getting rid of all the non-alphabet characters such as "!, &, #, ^, @ , ..." in the reference text that we are using to learn. Or by making all of the uppercase letters lowercase to have a more consistent text. I personally handled them where ever necessary.

# #### Part A: Initializations
# This includes importing needed libraries but more importantly assiging parameters for the genetic algorithm such as population size, number of individuals in the mating pool, number of genes that each chromosome has, the chance of crossover happening after being chosen, the rate of mutation. Stability interval will be explained later don't worry about it for now;) <br>
# <br>
# These quantities first came from either refrences or just putting a random number there:)) but improved with experimenting.
# 

# In[141]:


import sys, time
import string
import random
from nltk import ngrams
import collections
import re
import math
from heapq import heappop, heappush, nlargest

GLOBAL = 'global_text.txt'
POPULATION_SIZE = 60
MATING_POOL_SIZE = 15
GENES_COUNT = 26
N_GRAMS = 2
CROSSOVER_PROBABILITY = 0.55
CROSSOVER_POINTS = 3
MUTATION_RATE = 0.2
STABILITY_INTERVALS = 200


# #### Part B: Defining 
# Here <b>letters</b> are defined as <b>genes</b> building together a <b>key</b> which is our <b>chromosomes</b> for the genetic algorithm.

# In[142]:


alphabet = string.ascii_lowercase
print("The alphabet is:", alphabet)


# In[143]:


class Decode:
    def __init__(self, encoded_text):
        self.encoded_text = encoded_text
    def decode(self):
        decoded_text = ""
        key = genetic_algorithm()
        decoded_text = decrypt(encoded_text, key)
        return decoded_text, key


# Each <b>individual</b> as an object has a specific <b>chromosome</b> or <b>DNA</b> and a <b>fitness</b> value that will help us determine which individual is the <b>fittest</b> to be selected later by natural selection.

# In[144]:


class Individual:
    def __init__(self, chromosome, fitness):
        self.chromosome = chromosome
        self.fitness = fitness
    def __lt__(self, other):
        return self.fitness <= other.fitness


# #### Part C: n-Grams
# In order to understand our fitness function we must first define <b>n-Grams</b>. An n-Gram is a
# group of n letters which appear consecutively within a word. For that we use the ngrams method from NTLK library. We append each n-gram to a list to have them all together.

# In[145]:


def get_ngrams(text, n):
    grams = ngrams(text, n)
    allgrams = []
    for chars in grams:
        gram = ""
        for i in chars:
            gram += i
        allgrams.append(gram)
    return allgrams


# The reason n-Grams are so important is because their <b>frequency</b> for particular language can be determined by using a set of training texts and counting the <b>occurrences</b> of each n-Gram within the texts. The frequency of the n-Grams within a large enough sample of cipher text will be similar to the frequency deduced from the training texts. <br>
# Here we have <b>Global Text</b> as the "training" and <b>Encoded Text</b> as the "testing" texts.

# In[146]:


def count_ngrams(text):
    counter = collections.Counter()
    words = re.sub('[^{}]'.format(alphabet+alphabet.upper()), ' ', text).split()

    for word in words:
        for gram in get_ngrams(word, N_GRAMS):
            counter[gram] += 1

    return counter


# In[147]:


global_text = open(GLOBAL).read()
T = count_ngrams(global_text)    


# In[148]:


def get_letter():
    return random.choice(alphabet)


# In[149]:


def create_chromosome():
    chromosome = {}
    genes_count = 0
    while genes_count != GENES_COUNT:
        letter = get_letter()
        if letter not in chromosome:
            chromosome[letter] = alphabet[genes_count]
            genes_count += 1
        else:
            continue
        
    return chromosome


# In[150]:


def create_base_population(size):
    population = []
    for i in range(size):
        chromosome = create_chromosome()
        individual = Individual(chromosome, 0)
        heappush(population, individual)
    return population


# In[151]:


def decrypt(encoded, key):
    decoded = ''
    for char in encoded:
        if char.islower():
            decoded += key[char]
        elif key.get(char.lower()) != None:
                decoded += key.get(char.lower()).upper()
        else:
            decoded += char
    return decoded


# - choose n for all n-Grams
# - Let Frequency be defined as the number of occurrences of a n-Gram particular
# - Let T be the set of n-Grams found within the global text
# - Let N(x) be the set of n-Grams found within the decoded text by the chromosome x
# - Let $ F_{T}(y) $ = Frequency of y within global text if $ y \in T $ otherwise 0
# - Let $ F_{P}(x,y) $ = Frequency of y within decoded text by chromosome x if $ y \in N(x) $ otherwise 0
# - $ Fitness(x) = \sum_{y} F_{P}(x,y) \times log_2(F_{T}(y)) $

# In[152]:


def evaluate_fitness(chromosome, encoded_text):
    fitness = 0

    decoded = decrypt(encoded_text, chromosome)
    N_x = count_ngrams(decoded)
    
    for y in T:
        Ft = T[y]
        if y in N_x:
            Fp = N_x[y]
            fitness += Fp * Ft
        else:
            continue
    
    return fitness


# #### Part E: Selection
# - Sort the population by their fitnesses
# - Choose the best solutions from population for the mating pool. Even though the genetic algorithm says that each individual should have a chance regarding their fitness in practice I have notices that it will work much better if we just choose the best ones.
# - (Still if someone wants to do it by the books:<br>
#     First, definitely choose half the mating pool size the best solutions from population.
#     Second, choose the rest with weighted chance (weight is $\frac{individual's fitness}{\sum fitnesses}$))

# In[153]:


def select(population):
    
    selected_population = nlargest(int(MATING_POOL_SIZE), population)
   
#     fitness_list = []
#     for individual in population:
#         fitness_list.append(individual.fitness)
#     fitness_sum = sum(fitness_list)
    
#     chances = []
#     for individual in population:
#         chances.append(individual.fitness/fitness_sum)

#     for i in range(int(MATING_POOL_SIZE/2)):
#         selected_population.extend(random.choices(population, weights = chances))
        
    for i in selected_population:
        print(i.fitness, end = ' ')
    print()
    print()

    return selected_population


# #### Part F: Crossover
# - Given the two parents produced by the selection process,
# - Copy the mother genes into the child.
# - Choose some random letters, find the index of those letters in father genes and replace those child genes with father's.
# - We'll have half mother and half father genes in the end.

# In[154]:


def crossover(mother, father):
    mother_genes = []
    father_genes = []
    child_chromosome = {}
    child_genes = []
    times = int(GENES_COUNT/2)
    
    mother_genes.extend(list(mother.chromosome.keys()))
    father_genes.extend(list(father.chromosome.keys()))
    
    child_genes.extend(mother_genes)
    
    for i in range(times):
        char = random.choice(alphabet)
        index1 = father_genes.index(char) #index of j in father
        temp = child_genes[index1] #character in index -> e
        index2 = child_genes.index(char) #index of j in child at first
        child_genes[index1] = char #put j in index
        child_genes[index2] = temp
        
        
    chromosome_new = dict()
    
    for i in range(GENES_COUNT):
        chromosome_new[child_genes[i]] = alphabet[i]

    return chromosome_new


# #### Part G: Mutation
# Choose 2 random letters and swap them in the given chromosome. <br>
# <font color="29AB87"><b>What happens if we don't ever mutate after crossover? </b></font> Mutation is a random factor and and if it is never done, at the end we do not know whether we are actually closing in on the actual best solution, or whether we are stuck in a <b>locally</b> decent solution. In other words if a population evolves by only using the same pool of candidates, it is likely to become extremely homogeneous and <b>possibly miss out on better solutions.</b>

# In[155]:


def mutate(chromosome):
    keys = list(chromosome.keys())
    key1 = random.choice(keys)
    key2 = random.choice(keys)

    index1 = keys.index(key1)
    index2 = keys.index(key2)
    
    keys[index1] = key2
    keys[index2] = key1
    
    chromosome_new = dict()
    
    for i in range(GENES_COUNT):
        chromosome_new[keys[i]] = alphabet[i]

    return chromosome_new


# The similarity between crossover and mutation is that they both make changes to make a new individual for the new generation without the knowledge of what the final goal is. <font color="29AB87"><b>What is the difference between "crossover" and "mutation"?</b></font> The difference is that mutation is a totally random change to run away from being stuck with the same thing and not moving forward towards the goal but crossover happens with an algorithm to make a new chromosome from other chromosomes that were most likely picked for a reason. <br>
# <font color="29AB87"><b> Which is more effective in speeding up the process of reaching a better precision?</b></font> I would say crossover is more effective because it's not as random as mutation. There is always the possibily that mutation could take us further away from the goal. But crossover is more purposeful depending on your implementation.

# #### Part H: Genetic Algorithm
# 
# - Generate a random base population.
# - Evaluate the fitness of that population
# - Select top fittest of the population as part of the new generation (they stay alive and some of them might have children)
# - Randomly choose between the alive population(selected) as potential parents.
# - Crossover those parents with a particular chance to get a child.
# - Mutate the children with a particular chance.
# - Compute the fitness of the child and add it to the new generation.
# - Keep doing this until you reach the population size which means the population size is constant.<font color="29AB87"> <b>But what happens if the population kept getting larger?</b></font> Well it is obvious that calculating fitness for more individuals takes more time. If we make more and more individuals the bigger chance there is for that individual to be affectless in making a better population because we already have enough to get a reasonably correct answer.
# - Update the best fitness value anytime a new chromosome is to be added. 
# - <b>Stability Interval</b> is a pre-determined value that indicates how many more times should the program continue to search for a better answer if the best fitness has not changed hoping that it does at some point before counter reaches the stability interval. <font color="29AB87"><b>Why does this convergence to one value happen? What problems does this lead to? And what can we do about it?</b></font> My conclusion is that we get stuck in a local optimum. Even with mutation this can happen and we need to take a leap to look further by perhaps letting a random choromosome be added to the new generation or let some of the worst ones of the population get in the mating pool from time to time. Because sometimes the worst ones can have something that the best ones can't offer. For example, all of the best ones have gotten 2 letter is the key wrong but there is this one choromosome that has gotten almost everything wrong except for that 2 letters and that could literally save us. This can also be considered as a huge mutation that happens rarely.

# In[156]:


def genetic_algorithm():
    population = create_base_population(POPULATION_SIZE)
    best_fitness = 0
    iterations = 0
    last_fitness = 0
    last_fitness_increase = 0
    
    for individual in population:
        individual.fitness = evaluate_fitness(individual.chromosome, encoded_text)
        if individual.fitness > best_fitness:
            best_fitness = individual.fitness
            best_chromosome = individual.chromosome

    while last_fitness_increase < STABILITY_INTERVALS:
        #Selection;  //Natural Selection, survival of fittest
        population = select(population)
        
        #Crossover;  //Reproduction, propagate favorable characteristics
        while len(population) < POPULATION_SIZE: 
            potential_father = random.choice(population)
            potential_mother = random.choice(population)
            
            if random.choices([1,0], weights = [CROSSOVER_PROBABILITY, 1 - CROSSOVER_PROBABILITY]):
                child_chromosome = crossover(potential_mother, potential_father)
                child = Individual(child_chromosome, 0)
                
                #Mutation;
                if random.choices([1,0], weights = [CROSSOVER_PROBABILITY, 1 - CROSSOVER_PROBABILITY]):
                    child.chromosome = mutate(child.chromosome)

                child.fitness = evaluate_fitness(child.chromosome, encoded_text)
                if child.fitness > best_fitness:
                    best_fitness = child.fitness
                    best_chromosome = child.chromosome

                heappush(population, child)
            
        if best_fitness > last_fitness:
            last_fitness_increase = 0
            last_fitness = best_fitness
        else:
            last_fitness_increase += 1
        
        iterations += 1
    
    print('Best solution found after {} iterations:'.format(iterations))
    return best_chromosome


# In[157]:


encoded_text = open('encoded_text.txt').read()
d = Decode(encoded_text)
t1 = time.time()
decoded_text, key = d.decode()
print(decoded_text)
print(key)
t2 = time.time()
print(t2 - t1)


# #### References
# [1] Decrypting Substitution Ciphers with Genetic Algorithms, Jason Brownbridge, Department of Computer Science, University of Cape Town <br>
# [2] https://www.geeksforgeeks.org/genetic-algorithms/ <br>
# [3] https://towardsdatascience.com/introduction-to-genetic-algorithms-including-example-code-e396e98d8bf3<br>
# [4] https://towardsdatascience.com/genetic-algorithms-solving-ibms-january-2020-problem-e694d59f407d <br>

# In[ ]:




