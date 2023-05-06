# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 12:19:15 2023

@author: bened
"""
# a = line added by me

import sys,os,re
import fileinput,subprocess,inspect, random
import numpy as np
import matplotlib.pyplot as plt

##############################
#### Model Specifications ####
##############################

# I think in Perl Andy does define sth like empty arrays for both agent's spaces here
# and uses Perl's 'push' method to fill the space
#a, b = 'a','b' 
#agents=[a,b]
# We can't do it like this in Python but we can use a predefined np array of zeros, see below


# num of word categories
## 4 square
start = [[35, 35], [35, 65], [65, 35], [65, 65]]

## 25
#start=[[18,18],[36,18],[50,18],[68,18],[86,18],[18,36],[36,36],[50,36],[68,36],[86,36],
#       [18,50],[36,50],[50,50],[68,50],[86,50],[18,68],[36,68],[50,68],[68,68],[86,68],
#       [18,86],[36,86],[50,86],[68,86],[86,86]] # len 25, in Perl @ to indicate array

lexemes=start

## Whether or not the initial exemplar positions are all random
randstart=0

## How many 'segments' a word has
wordlength=2

## size of the field
dimension=100

## number of exemplars for each word
exnum=100

splitdemo=0 # ???

# stdev is used in initializing the exemplar clouds to determine how wide the cloud is
stdev=3.8


# Unclear why these numbers were chosen
strongbias= 0.5 # ???

biasfact=1000000 # ???


# This isn't a traditional decay formula 
decay=exnum/5 # ??? will be 20 when doing 100 exemplars




## Factor determining the range over which attraction/reversion occurs

## W = word level, S = sound level

# scaling factor determining the rate at which influence from other exemplars falls of with distance; larger value means a narrower range
Wspread=-0.2

Sspread=-0.2

## Scaling factor determining the range over which category similarity is calculated
catfactor=-0.2





## Settings for whether particular parts of the simulation are operative

soundreversion=1 # ???

wordreversion=1 # ???

catcomp=1 # == anti ambiguity bias

lexdiffusion=0 # ???


##############################################################################################################
##############################################################################################################



###################
#### Functions ####
###################

##############################################################################################################

#######################
#### Choose output ####
#######################
def chooseoutput(speaker, i):
    """
    Variables utilized within this:
    outword
    wordreversion
    speaker
    splitdemo
    Wspread
    totsim
    soundreversion
    Sspread
    wordlength
    decay
    
    Contingencies:
    None

    Function output:
    outword

    Effect:
    To take the speaker and to edit the outword based on either word based shift or sound based shift. 

    """

    word_cat = speaker[i] # a
    
    inproportions=[] # == activations
    proptotal=0
    
    # $i is the lexeme being produced, inherited from the trunk loop above
    #for j(0 +  + (len()-1)[[speaker][i]]
    for j in range(1,exnum+1):
       #gets the position (=activation) of every exemplar, puts it into a list
       prop=2.72**(-j/decay) # decay is 20
       inproportions.append(prop)
       proptotal+=prop
    #print(inproportions)
     
    #chance=0
    #randpick= random.uniform(0, 1)
    chances = list() # a
    for d in range(0,len(inproportions)):
        chance = inproportions[d]/proptotal
        #centerpoint = d
        chances.append(chance) # a

    cp_pos = np.random.choice(np.arange(100), 1, p=chances) # a choosing index based on chance which is based on activation
    centerpoint = word_cat[cp_pos][0]  # a
    #print(centerpoint)
    #print(cp_pos)

# Now have a centerpoint in the lexical category. 
# Make a shifted version based on the existing word and sound exemplars

# Word-based shift
# Calculate summed euclidean distance of each exemplar in the word category to the centerpoint exemplar. 

    outword=[0,0]
    for k in range(0,wordlength): #each segment   
        totsim=0
        for j in range(0,exnum): #each exemplar
            exdist=abs((speaker[i][cp_pos]).flatten()[k]-(speaker[i][j]).flatten()[k])            
            sim=(2.72**(Wspread*exdist))*(2.72**(-j/decay))
            outword[k]+=speaker[i][j][k]*sim
            totsim+=sim

        outword[k] /= totsim #FAIL $outword[$k]/=$totsim;

    #print('outword ',outword)
    
# Sound-based shift
    if soundreversion==1:
        outsounds=[0,0]
        
        for k in range(0,wordlength): #each segment
            totsim=0
            for h in range(0,len(lexemes)): #each lexical category
              for j in range(0,exnum): #each exemplar                    
                    exdist=abs((speaker[i][cp_pos]).flatten()[k]-(speaker[h][j]).flatten()[k])
                    sim=(2.72**(Sspread*exdist))*(2.72**(-j/decay))
                    outsounds[k]+=speaker[h][j][k]*sim
                    totsim+=sim
            outsounds[k] /= totsim
            
        #print('outsounds ',outsounds)


#merging the two

# The relative contribution to the output segment value from the word category is .9 vs .1 for the sound category

    if wordreversion==0:
        pass # a | not needed since defult is 1
    elif soundreversion==0:
        pass # a | not needed since defult is 1
    else:
        for a in range(0,len(outword)):
          outword[a]=((9*outword[a])+outsounds[a])/10
    #print('outword ',outword)
    return outword
    
##############################################################################################################

###############
#### Noise ####
###############

def noise(outword):
    """
    Variables utilized in this:
    outword
    dimension
    splitdemo
    strongbias
    rn
    lexdiffusion
    biasfact
    err
    stdev
    cyc

    Contingencies:
    chooseoutput

    Function output:
    outword

    Effect:
    To add noise to the outword.

    """

    for a in range(0,len(outword)):
      rn=0
      for i in range(20):
          rn+= random.uniform(0, 1) - 0.5
      print('rn ',rn)    
      err=rn*stdev
    
      bias=((outword[a]-(dimension/2))**2)/(biasfact*strongbias)
      print('bias ', bias)
      
      if outword[a]>(dimension/2):
           bias*=-1  
      outword[a]=int(outword[a]+err+bias+0.5)
      if lexdiffusion:
          pass # a 
      if splitdemo:
          pass
    
      if outword[a]<0:
           outword[a]=0
      if outword[a]>dimension:
           outword[a]=dimension
    print(outword)
    return outword

##############################################################################################################

########################
#### Categorization ####
########################

def categorization(outword):
   #global outword,exnum,catcomp,speaker,catfactor,excluded,totsim,listener,decay,

   totsim=0
   condprobs=[]

   for q in range(0,len(lexemes)): # each category
      catsim=0
      for r in range(0,exnum): #each exemplar
         sumsqdiff=0
         for s in range(0,wordlength): #each segment
            sumsqdiff+=(outword[s]-listener[q][r][s])**2
         sumsqdiff=sumsqdiff**0.5
         catsim+=2.72**(catfactor*sumsqdiff)*2.72**(-r/decay) # catfactor == -0.2
      totsim+=catsim
      condprobs.append(catsim)
      print('Catsim of outword to cat ',q,' ',catsim)
   for u in range(0,len(condprobs)): #normalizing similarity to the total
      condprobs[u]=condprobs[u]/totsim
   print("Condprobs ",condprobs)
   print("Condprobs sum",sum(condprobs))

   choice=0
   for c in range(0,len(condprobs)):
       if condprobs[c]>condprobs[choice]: #if the conditional probability to this category is greater than to the prior, set 'choice' to that list position
          choice=c
   print("choice: ", choice)
 
   ### Negative selection
   if catcomp==1:
       if random.uniform(0, 1)<condprobs[choice]: # if rand(1) is less than the similarity between the two categories, store the exemplar
           listener[choice] = np.reshape(np.insert(listener[choice][0:-1],0,outword),(100,-1))
           # Various other ways of setting the threshold for successful storage
           #if ($condprobs[$choice]< .9) {
           #if (rand(1) > $condprobs[$choice]**2) 

   else: # store always
       listener[choice] = np.reshape(np.insert(listener[choice][0:-1],0,outword),(100,-1))
     


# This is our plotting function
def plot_space(agent,cycle): 
    fig, ax = plt.subplots()
    plt.xlim([0, 100])
    plt.ylim([0, 100])
    colors = ['r','g','b','y']
    markers = ["v" , "o" , "," , "x"]
    for i in range(4):
        x = agent[i,:,0].flatten()
        y = agent[i,:,1].flatten()
        ax.scatter(x,y, color=colors[i],marker=markers[i])
    name = 'Listener_Cycle_'+str(cycle)
    plt.title(name)
    if cycle%1000 == 0:
        plt.savefig('Output/'+name+'.png')
    plt.show()





##############################################################################################################
##############################################################################################################


####################
#### Simulation ####
####################

# Create empty agent's spaces here
a = np.zeros((len(lexemes),dimension,wordlength))
b = np.zeros((len(lexemes),dimension,wordlength))
agents=[a,b]

batchnum=2

for batch in range(0,batchnum):
    #print('batch: ',batch)
    #a=[] # @ is used for array
    #b=[] # @ is used for array
      
     
    ## Initializing the exemplar clouds for each lexeme
    
    for ag in agents:
       for i in range(0,len(lexemes)):
          for j in range(0 ,100):
             for k in range(0,wordlength):
                rn=0
                for n in range (0,20):
                   rn += random.uniform(0, 1) - 0.5 # rand(1) in Perl generates rand number between 0 and 1
                err=rn*stdev
                startex = start[i][k]+int(err+0.5)
                #print(startex)
    
                if k>1:
                   k-=2
                if randstart:
                   startex=int(random.rand(dimension))
                if startex<0:
                   startex=0
                if startex>dimension:
                   startex=dimension
                ag[i,j,k] = startex #a
               
               
    cyc=1  # ???
    avglist=[]  # ???
    #&scatter;
    #&print;
    
    exrun=0  # ???
    exruntot=0  # ???
    Csdrun=0  # ???
    Csdruntot=0  # ???
    Csdavgtot=0  # ???

listener=agents[random.randint(0,1)]


# number of cycles
cyctot=4001

## Production-perception loop
for cyc in range(0,cyctot):
    if cyc%25 == 0:
        plot_space(listener, cyc)
    for speaker in agents:
        listener=agents[random.randint(0,1)] # Andy randomly chooses the listener
        
        for i in range(0,len(lexemes)):  # i is the lexeme being produced == word cat
           categorization(noise(chooseoutput(speaker,i)))
           