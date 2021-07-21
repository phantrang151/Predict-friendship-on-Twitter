# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 20:11:55 2019

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
#train_graph_undirected = nx.read_edgelist('train_pos_x_500000.csv',delimiter=',',create_using=nx.Graph(),nodetype=int)

# data for training includes both positive and negative edges
df_final_train = pd.read_csv('train_x_200K.csv', names=['source', 'target'], header=None) 

'''
---------------------Support functions to create features----------------------
'''
    
# follow back    
def follows_back(a,b):
    if train_graph.has_edge(b,a): return 1
    else: return 0



'''
---------------------------Build Features--------------------------------------
'''

#mapping followback or not on train
df_final_train['follows_back'] = df_final_train.apply(lambda row: follows_back(row['source'],row['target']),axis=1)

pd.DataFrame(df_final_train['follows_back']).to_pickle("train_features_follows_back.pkl")

#------------------------------------------------------------------------------








