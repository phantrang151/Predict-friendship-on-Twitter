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
from scipy.special import comb


'''Step 1
-------------------Change directory -------------------------------------------
'''
path = pathlib.Path(r"C:\Users\Trang\Master of Business Analytics\Module 3\Machine Learning\Project\Codes\Template Code\Codes for Exploratory Analysis")
os.chdir(path)

'''Step 2
-------------------Place train_pos_x here -------------------------------------
'''
train_graph = nx.read_edgelist('TOTAL_EDGES.csv',delimiter=',',create_using=nx.DiGraph(),nodetype=int)
#train_graph_undirected = nx.read_edgelist('TOTAL_EDGES.csv',delimiter=',',create_using=nx.Graph(),nodetype=int)


'''
---------------------Exploratory Analysis--------------------------------------
'''

'''
----------------Point 1: "There is a low chance of following ------------------
'''
# on average, one person follows 5 people
# and there are 5 milliion nodes
# the chance A follows B is low
print(nx.info(train_graph))

'''
Number of nodes: 4867136
Number of edges: 23946602
Average in degree:   4.9201
Average out degree:   4.9201
'''

'''
---------------------Point 2: The graphs is sparse-----------------------------
'''
# so we can generate negative edges randomly,low chance of being positive
total_nodes = nx.number_of_nodes(train_graph)
total_present_edges = nx.number_of_edges(train_graph)

total_possible_edges = comb(total_nodes, 2, exact=False)

number_missing_edges = total_possible_edges - total_present_edges

print("Total missing edges: ", number_missing_edges)
print("Percentage of actual edges/ missing edges: ",total_present_edges/number_missing_edges)
'''
Total missing edges:  11844480041078.0
Percentage of actual edges/ missing edges: 2.021752066528076e-06
'''

'''
--------------Point 3: Test nodes are in train nodes---------------------------
'''
test_nodes = set()
reader = open('TEST_NODES.csv','r')
lines = reader.readlines()

for line in lines:
    line = line.strip()
    test_nodes.add(int(line)) 

# print if the test_nodes is the subset of total nodes
is_subset = test_nodes.issubset(set(train_graph.nodes()))
print('If test nodes is a subset of total nodes: ',is_subset) # return true


'''
----------Point 4: Draw the distribution of incoming and outgoing edges--------

'''
in_degree_list = []
no_follower_count = 0
total_node_list = train_graph.nodes()
for node in total_node_list:
    num_of_follower = train_graph.in_degree(node)
    in_degree_list.append(num_of_follower)
    if num_of_follower==0 :
        no_follower_count += 1
        
in_degree_list.sort()       
print('number of people not followed by anyone: ',no_follower_count) # return 0
plt.plot(in_degree_list)  

out_degree_list = []
no_followee_count = 0
for node in total_node_list:
    num_of_followee = train_graph.out_degree(node)
    out_degree_list.append(num_of_followee)
    if num_of_followee==0 :
        no_followee_count += 1
        
out_degree_list.sort()
print('number of people not following anyone: ',no_followee_count) # 4847566
plt.plot(out_degree_list) 