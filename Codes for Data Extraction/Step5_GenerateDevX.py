# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 18:13:53 2019

@author: Shi Song
"""
import networkx as nx
import pandas as pd
import os,pathlib
import random

'''
Need fix this errors
line 34, in <module>
edges[(edge[0], edge[1])] = 1

TypeError: 'int' object is not subscriptable
'''


#Need change the dir
path = pathlib.Path(r"C:\Users\Trang\Master of Business Analytics\Module 3\Machine Learning\Project\Codes\Template Code\Codes for Data Extraction")
os.chdir(path)



#Positive edges 24 million
train_pos_x_24M = pd.read_csv("TOTAL_EDGES.csv",header=None)
train_pos_x_24M = pd.DataFrame(train_pos_x_24M)

#Set the number of missing edges/positive edges you want to generate
numberOfEdges = 25000

#Random selecte the positive edges and save to a csv file
dev_pos_25K = train_pos_x_24M.sample(n=numberOfEdges, random_state=9999)
dev_pos_25K = pd.DataFrame(dev_pos_25K)
dev_pos_25K.to_csv('dev_pos_x_25K.csv',header=False,index=False)

#Generating nodes list from given graph
g = nx.read_edgelist('TOTAL_EDGES.csv',delimiter=',',create_using=nx.DiGraph(),nodetype=int)
list_of_postive_nodes = list(g.nodes)

#the dict will contain a tuple of 2 nodes as key and the value will be 1 if the nodes are connected else -1
edges = dict()
for index, edge in train_pos_x_24M.iterrows():
    edges[(edge[0],edge[1])] = 1

    
dev_neg_x_25K = set([])
while (len(dev_neg_x_25K) < numberOfEdges):
	a=random.choice(list_of_postive_nodes)
	b=random.choice(list_of_postive_nodes)
	tmp = edges.get((a,b),-1)
	if tmp == -1 and a!=b:
		try:
            # adding points who less likely to be friends
			if nx.shortest_path_length(g,source=a,target=b) > 2: 

				dev_neg_x_25K.add((a,b))
			else:
				continue  
		except:  
				dev_neg_x_25K.add((a,b))              
	else:
		continue
#Saving to a csv file
dev_neg_x_25K = pd.DataFrame(dev_neg_x_25K)
dev_neg_x_25K.to_csv('dev_neg_x_25K.csv',header=False, index=False)

#Combin the positive and negtive together
dev_x_50K = dev_neg_x_25K.append(dev_pos_25K,ignore_index=True)
dev_x_50K.to_csv('dev_x_50K.csv',header=False,index=False)

