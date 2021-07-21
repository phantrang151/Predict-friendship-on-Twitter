# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 20:07:16 2019

@author: Trang
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 17:11:27 2019

@author: Trang
"""
#%reset -f

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
#train_graph = nx.read_edgelist('train_24M.csv',delimiter=',',create_using=nx.DiGraph(),nodetype=int)
train_graph_undirected = nx.read_edgelist('TOTAL_EDGES.csv',delimiter=',',create_using=nx.Graph(),nodetype=int)

# data for training includes both positive and negative edges
df_final_train = pd.read_csv('train_x_200K.csv', names=['source', 'target'], header=None) 

'''
---------------------Support functions to create features----------------------
'''
 

def resource_allocation(a,b):

    iterator_coefs = nx.resource_allocation_index(train_graph_undirected,[(a, b)])
    coeff = []
    for u, v, p in iterator_coefs:
        coeff.append(p)
    return (p)




'''
---------------------------Build Features--------------------------------------
'''

df_final_train['resource_allocation'] = df_final_train.apply(lambda row:resource_allocation( row['source'], row['target']),axis=1)
#df_final_dev['jaccard'] = df_final_dev.apply(lambda row:jaccard(row['source'],row['target']),axis=1)

# export the columns to pickle files and drop the column
pd.DataFrame(df_final_train['resource_allocation']).to_pickle("train_features_resource_allocation.pkl")

#construct the data set for machine learning

#------------------------------------------------------------------------------








