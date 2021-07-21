# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 20:08:45 2019

@author: Trang
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 17:11:27 2019

@author: Trang
"""
import math
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from time import time
import csv
import pickle
import os,pathlib
from tqdm import tqdm
from pandas import HDFStore, DataFrame
from pandas import read_hdf

path = pathlib.Path(r"C:\Users\Trang\Master of Business Analytics\Module 3\Machine Learning\Project\Version4 - Random 100K - 24M Edges")
os.chdir(path)

# build the graph on positive data
train_graph = nx.read_edgelist('TOTAL_EDGES.csv',delimiter=',',create_using=nx.DiGraph(),nodetype=int)
#train_graph_undirected = nx.read_edgelist('train_24M.csv',delimiter=',',create_using=nx.Graph(),nodetype=int)

# data for training includes both positive and negative edges
df_final_train = pd.read_csv('train_x_200K.csv', names=['source', 'target'], header=None) 

'''
---------------------Support functions to create features----------------------
'''
  

# If has direct edge, then delete that edge and calculate shortest path
def compute_shortest_path_length(a,b):

    shortest_path = nx.shortest_path_length(train_graph,source=a,target=b)
    return shortest_path



'''
---------------------------Build Features--------------------------------------
'''
#mapping shortest path on train 
df_final_train['shortest_path'] = df_final_train.apply(lambda row: compute_shortest_path_length(row['source'],row['target']),axis=1)

pd.DataFrame(df_final_train['shortest_path']).to_pickle("train_features_shortest_path.pkl")


#------------------------------------------------------------------------------








