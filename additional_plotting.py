# -*- coding: utf-8 -*-
"""
Created on Tue May  2 21:29:17 2023

@author: bened
"""

import numpy as np
import matplotlib.pyplot as plt

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
    plt.savefig('Output/'+name+'.png')
    #plt.show()
    
start = np.array([[[35, 35]], [[35, 65]], [[65, 35]], [[65, 65]]])

plot_space(start, 0)