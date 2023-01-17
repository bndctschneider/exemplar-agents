# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 08:17:34 2023

@author: bened
"""

import numpy as np
from numpy import dot
from numpy.linalg import norm
import matplotlib.pyplot as plt
import random

# To-Do
# find out production/perception order (paper) -> last sentence p.37
# investigate why last cat almost never chosen in perception SIMILARITY METRIC?
# how is exemplar space seeded
# Understand population vectors for similarity bias



def cos_sim(a,b):
    if a.all()==0 or b.all()==0: # ensuring no division by 0
        cos_sim = -1
    else:
        cos_sim = dot(a, b)/(norm(a)*norm(b))
    return cos_sim

def insert_word(space,word_cat,word):
    '''
    Insert word in specified word_category in exemplar space.
    Insert at first position, delete last.
    '''
    new_wc = np.insert(space[word_cat][0:-1],0,word) # create new word cat array, shape (200,)
    new_wc = np.reshape(new_wc,(100,-1)) # reshape as (100,2)
    space[word_cat] = new_wc
    return space

class Agent:
    def __init__(self,name):
        
        self.name = name
        
        #### Initialize space ####
        self.space = np.zeros((4,100,2))
        
        #### Initialize word categories ####
        # init word cat 1 == 'ba'
        #self.space = insert_word(self.space, 0, [26,26])
        #self.space = insert_word(self.space, 0, [25,25])
        self.space = insert_word(self.space, 0, [2,2])
        self.space = insert_word(self.space, 0, [1,1])
        
        # init word cat 2 == 'pa'
        #self.space = insert_word(self.space, 1, [76,26])
        #self.space = insert_word(self.space, 1, [75,25])
        self.space = insert_word(self.space, 1, [99,2])
        self.space = insert_word(self.space, 1, [98,1])
        
        # init word cat 3 == 'bi'
        #self.space = insert_word(self.space, 2, [26,76])
        #self.space = insert_word(self.space, 2, [25,75])
        self.space = insert_word(self.space, 2, [2,99])
        self.space = insert_word(self.space, 2, [1,98])
        
        # init word cat 4 == 'pi'
        #self.space = insert_word(self.space, 3, [76,76])
        #self.space = insert_word(self.space, 3, [75,75])
        self.space = insert_word(self.space, 3, [99,99])
        self.space = insert_word(self.space, 3, [98,98])
        
        self.words = np.reshape(self.space,(self.space.shape[0]*self.space.shape[1],-1))
        self.segments_dim1 = self.space[:,:,0].flatten()
        self.segments_dim2 = self.space[:,:,1].flatten()
        
        # self.space:
        # axis 0 = word cats
        # axis 1 = words in each cat
        # axis 2 = segments
        
        #### Activations ####
        positions = np.arange(1,101,1,dtype=int)
        self.activations = np.exp(-0.2*positions)
        
    def choose_rword(self,wc_index):
        #wc_index = random.randint(0, 3)# randomly select word cat index
        #print('Produced cat: ',wc_index)
        word_cat = self.space[wc_index] # select word cat to pick word from
        choice_probs = self.activations/self.activations.sum() # turning activations into prob distr
        random_word_idx = np.random.choice(len(word_cat),1,p=choice_probs) # sample index given prob distr
        random_word = word_cat[random_word_idx] # pick word from chosen word cat with chosen index
        #print('random word: ', random_word)
        
        # Ensure that no exemplar [0,0] is picked
        if random_word.all()!=0:
            #print('not all 0')
            return random_word.flatten()
        else:
            #print('all 0')
            return self.choose_rword(wc_index)
                
    def produce(self,wc_index):
        random_word = self.choose_rword(wc_index)
        noise = np.random.normal(0,3,1) # Random noise, mean 0 std 3, used during output creation
        noise = np.absolute(noise)
        rw_noise = np.where(random_word<50,random_word+noise,random_word-noise) # if smaller 50, add noise, else substract
        return rw_noise
    
    def receive(self,word):
        max_sim = 0
        max_ind = None
        #print('Recieved word: ',word)
        for ind, word_cat in enumerate(self.space):
            #print('ind ',ind)
            for w in word_cat:
                sim = cos_sim(word, w)
                if sim > max_sim and not np.array_equal(word, w):
                    max_word = w
                    max_sim = sim
                    max_ind = ind
        #print('Highest sim to: ',max_word)
        #print('Store in word category: ',self.space[max_ind])
        #print('max ind',max_ind)
        self.space = insert_word(self.space, max_ind, word)
        #print(self.space)
                
    def plot_space(self):
        x = self.space[:,:,0].flatten()
        x = x[x!=0] # remove 0s for plotting
        y = self.space[:,:,1].flatten()
        y= y[y!=0] # remove 0s for plotting
        plt.scatter(x,y)
        plt.title(self.name)
        plt.show()
        
    
        
def pp_loop(iterations,agent1,agent2):
    for i in range(iterations):
        # Agent 1 produces, agent 2 receives
        agent2.receive(agent1.produce(0))
        agent2.receive(agent1.produce(1))
        agent2.receive(agent1.produce(2))
        agent2.receive(agent1.produce(3))
        
        # Agent 2 produces, agent 1 receives
        agent1.receive(agent2.produce(0))
        agent1.receive(agent2.produce(1))
        agent1.receive(agent2.produce(2))
        agent1.receive(agent2.produce(3))
        
        agent2.plot_space()
    print(agent1.space)
    
    
        
if __name__=='__main__':
    agent1 = Agent('Agent 1')
    agent2 = Agent('Agent 2')
    pp_loop(100,agent1,agent2)

