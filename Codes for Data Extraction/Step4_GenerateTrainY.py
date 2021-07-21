import pandas as pd
import numpy as np
import os,pathlib

path = pathlib.Path(r"C:\Users\Trang\Master of Business Analytics\Module 3\Machine Learning\Project\Codes\Template Code\Codes for Data Extraction")
os.chdir(path)

numberOfEdges = 100000

# step 1
Y_train_positive = []
for i in range(numberOfEdges):
    Y_train_positive.append(1)
    
positive_frame_y = pd.DataFrame(Y_train_positive)
positive_frame_y.to_csv("train_pos_y_100K.csv", encoding='utf-8', index=False, header=None)

# step 2
Y_train_negative = []
for i in range(numberOfEdges):
    Y_train_negative.append(0)
    
negative_frame_y = pd.DataFrame(Y_train_negative)
negative_frame_y.to_csv("train_neg_y_100K.csv", encoding='utf-8', index=False, header=None)

# step 3

train_y = Y_train_negative + Y_train_positive

train_y_frame = pd.DataFrame(train_y)
train_y_frame.to_csv("train_y_200K.csv", encoding='utf-8', index=False, header=None)
