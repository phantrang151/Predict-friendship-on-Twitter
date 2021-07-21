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
 
def in_degree(a,b):

    in_degree_b = train_graph.in_degree(b)
    return math.log2(in_degree_b + 1)

def out_degree(a,b):

    out_degree_a = train_graph.out_degree(a)
    return math.log2(out_degree_a + 1)

'''
---------------------------Build Features--------------------------------------
'''
df_final_train['in_degree'] = df_final_train.apply(lambda row:in_degree( row['source'], row['target']),axis=1)
# export the columns to pickle files and drop the column
pd.DataFrame(df_final_train['in_degree']).to_pickle("train_features_in_degree.pkl")
df_final_train.drop(['in_degree'], axis=1, inplace=True) 

df_final_train['out_degree'] = df_final_train.apply(lambda row:out_degree( row['source'], row['target']),axis=1)
# export the columns to pickle files and drop the column
pd.DataFrame(df_final_train['out_degree']).to_pickle("train_features_out_degree.pkl")
df_final_train.drop(['out_degree'], axis=1, inplace=True) 
#------------------------------------------------------------------------------








