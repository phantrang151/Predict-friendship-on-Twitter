# -*- coding: utf-8 -*-
"""
--------------------------  4 steps -------------------------------------------
"""
from time import time
import math
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

import csv
import pickle
import os,pathlib
from tqdm import tqdm
from pandas import HDFStore, DataFrame
from pandas import read_hdf

start = time()
'''Step 1
-------------------Change directory -------------------------------------------
'''
path = pathlib.Path(r"C:\Users\Trang\Master of Business Analytics\Module 3\Machine Learning\Project\Version4 - Random 100K - 24M Edges")
os.chdir(path)

'''Step 2
-------------------Place train_pos_x here -------------------------------------
'''
train_graph = nx.read_edgelist('TOTAL_EDGES.csv',delimiter=',',create_using=nx.DiGraph(),nodetype=int)
train_graph_undirected = nx.read_edgelist('TOTAL_EDGES.csv',delimiter=',',create_using=nx.Graph(),nodetype=int)

'''Step 3
-------------------Place train_x here -----------------------------------------
'''
df_final_train = pd.read_csv('train_x_200K.csv', names=['source', 'target'], header=None) 

'''
---------------------Support functions to create features----------------------
'''
def jaccard(a,b):

    iterator_coefs = nx.jaccard_coefficient(train_graph_undirected,[(a, b)])
    coeff = []
    for u, v, p in iterator_coefs:
        coeff.append(p)
    return (p)

def resource_allocation(a,b):

    iterator_coefs = nx.resource_allocation_index(train_graph_undirected,[(a, b)])
    coeff = []
    for u, v, p in iterator_coefs:
        coeff.append(p)
    return (p)
    

# If has direct edge, then delete that edge and calculate shortest path
def compute_shortest_path_length(a,b):

    shortest_path = nx.shortest_path_length(train_graph,source=a,target=b)
    return shortest_path
    
# follow back    
def follows_back(a,b):
    if train_graph.has_edge(b,a): return 1
    else: return 0

def in_degree(a,b):

    in_degree_b = train_graph.in_degree(b)
    return math.log2(in_degree_b + 1)

def out_degree(a,b):

    out_degree_a = train_graph.out_degree(a)
    return math.log2(out_degree_a + 1)

def adar(a,b):

    iterator_coefs = nx.adamic_adar_index(train_graph_undirected,[(a, b)])
    coeff = []
    for u, v, p in iterator_coefs:
        coeff.append(p)
    return (p)


def preferential_attachment(a,b):

    iterator_coefs = nx.preferential_attachment(train_graph_undirected,[(a, b)])
    coeff = []
    for u, v, p in iterator_coefs:
        coeff.append(p)
    return (p)

'''
---------------------------Build Features--------------------------------------
'''
#mapping jaccrd followers to train and test data
df_final_train['jaccard'] = df_final_train.apply(lambda row:jaccard( row['source'], row['target']),axis=1)
pd.DataFrame(df_final_train['jaccard']).to_pickle("train_features_jaccard.pkl")
df_final_train.drop(['jaccard'], axis=1, inplace=True) 


df_final_train['resource_allocation'] = df_final_train.apply(lambda row:resource_allocation( row['source'], row['target']),axis=1)
pd.DataFrame(df_final_train['resource_allocation']).to_pickle("train_features_resource_allocation.pkl")
df_final_train.drop(['resource_allocation'], axis=1, inplace=True) 

#mapping shortest path on train 
df_final_train['shortest_path'] = df_final_train.apply(lambda row: compute_shortest_path_length(row['source'],row['target']),axis=1)
pd.DataFrame(df_final_train['shortest_path']).to_pickle("train_features_shortest_path.pkl")
df_final_train.drop(['shortest_path'], axis=1, inplace=True) 

#mapping followback or not on train
df_final_train['follows_back'] = df_final_train.apply(lambda row: follows_back(row['source'],row['target']),axis=1)
pd.DataFrame(df_final_train['follows_back']).to_pickle("train_features_follows_back.pkl")
df_final_train.drop(['follows_back'], axis=1, inplace=True) 

df_final_train['in_degree'] = df_final_train.apply(lambda row:in_degree( row['source'], row['target']),axis=1)
pd.DataFrame(df_final_train['in_degree']).to_pickle("train_features_in_degree.pkl")
df_final_train.drop(['in_degree'], axis=1, inplace=True) 


df_final_train['out_degree'] = df_final_train.apply(lambda row:out_degree( row['source'], row['target']),axis=1)
pd.DataFrame(df_final_train['out_degree']).to_pickle("train_features_out_degree.pkl")
df_final_train.drop(['out_degree'], axis=1, inplace=True) 


df_final_train['preferential_attachment'] = df_final_train.apply(lambda row:preferential_attachment( row['source'], row['target']),axis=1)
pd.DataFrame(df_final_train['preferential_attachment']).to_pickle("train_features_preferential_attachment.pkl")
df_final_train.drop(['preferential_attachment'], axis=1, inplace=True) 


df_final_train['adar'] = df_final_train.apply(lambda row:adar( row['source'], row['target']),axis=1)
pd.DataFrame(df_final_train['adar']).to_pickle("train_features_adar.pkl")
df_final_train.drop(['adar'], axis=1, inplace=True) 

#construct the data set for machine learning
jaccard_df = pd.read_pickle("train_features_jaccard.pkl")
shortest_path_df = pd.read_pickle("train_features_shortest_path.pkl")
follows_back_df = pd.read_pickle("train_features_follows_back.pkl")
resource_allocation_df = pd.read_pickle("train_features_resource_allocation.pkl")
in_df = pd.read_pickle("train_features_in_degree.pkl")
out_df = pd.read_pickle("train_features_out_degree.pkl")
adar_df = pd.read_pickle("train_features_adar.pkl")
referential_attachment_df = pd.read_pickle("train_features_referential_attachment.pkl")

''' Step 4
------------------- Change your file name -------------------------------------
'''
final_data_forML =pd.concat([jaccard_df,resource_allocation_df, shortest_path_df,follows_back_df,in_df,out_df,adar_df,referential_attachment_df], axis=1)
final_data_forML.to_pickle("train_x_ML_200K.pkl")

# careful ! this data is big
#test_final_data_forML = pd.read_pickle("final_data_forML.pkl")
end = time()
print("Total time",(end - start)/60)









