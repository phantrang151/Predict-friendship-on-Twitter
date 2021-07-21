import numpy as np  
import matplotlib.pyplot as plt  
import pandas as pd 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import train_test_split

X_test = []
test_nodes = []    
    
test_text = open('test-public.txt','r')
test_lines = test_text.readlines()

# skip the header
test_lines = test_lines[1:]

for line in test_lines:
    line = line.strip()
    pair = line.split('\t')
    X_test.append([pair[1],pair[2]]) # just keep the node, skip the node id
    
    if pair[1] not in test_nodes:
        test_nodes.append(pair[1])
    if pair[2] not in test_nodes:
        test_nodes.append(pair[2])
        
test_nodes_frame = pd.DataFrame(test_nodes)
test_nodes_frame.to_csv("TEST_NODES.csv", encoding='utf-8', index=False, header=None)

X_test_frame = pd.DataFrame(X_test)
X_test_frame.to_csv("TEST_EDGES.csv", encoding='utf-8', index=False, header=None)


