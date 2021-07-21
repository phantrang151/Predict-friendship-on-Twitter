# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 20:13:41 2019

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







