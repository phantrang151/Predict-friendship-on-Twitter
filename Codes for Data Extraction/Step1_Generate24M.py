# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 00:27:18 2019

@author: Trang
"""


import numpy as np  
import matplotlib.pyplot as plt  
import pandas as pd 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import train_test_split
import csv
import random


# preparing training data
X_train = []
X_nodes = set()

reader = open('train.txt','r')
lines = reader.readlines()

for line in lines:
    line = line.strip()
    nodes = line.split('\t')
    #assert(len(nodes)>=2)
    if nodes[0] not in X_nodes:
            X_nodes.add(nodes[0]) 
            
    for y in range(1,len(nodes)):
        X_train.append([nodes[0],nodes[y]])
        if nodes[y] not in X_nodes:
            X_nodes.add(nodes[y]) 
        

X_total_edges_frame = pd.DataFrame(X_train)
X_total_edges_frame.to_csv("TOTAL_EDGES.csv", encoding='utf-8', index=False, header=None)

X_total_nodes_frame = pd.DataFrame(list(X_nodes))
X_total_nodes_frame.to_csv("TOTAL_NODES.csv", encoding='utf-8', index=False, header=None)