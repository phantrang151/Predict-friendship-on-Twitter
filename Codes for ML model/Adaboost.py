''' Step 0
--------------------Reset parameter & import required package  ----------------------
'''
%reset -f

import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import csv
import os,pathlib

''' Step 1
--------------------Change configurations & file names  ----------------------
'''
path = pathlib.Path(r"C:\Users\Billi\Documents\2019\Module 3\Machine Learning\Assignment")
os.chdir(path)

X_train = pd.read_pickle('./pkl/train_x_ML_200K.pkl')
Y_train = pd.read_csv('train_y_200K.csv',  header=None)
Y_train = np.array(Y_train[0]) # convert to array, using only the first column

X_dev = pd.read_pickle('./pkl/dev_x_ML_50K.pkl')
Y_dev = pd.read_csv('dev_y_50K.csv',  header=None)
Y_dev = np.array(Y_dev[0])

X_test = pd.read_pickle('./pkl/test_x_ML_2K.pkl')

''' Step 2(Optional)
--------------------Drop the features  ------------------------------------
'''
# FULL LIST:
# ['jaccard', 'resource_allocation', 'shortest_path', 'follows_back','in_degree', 'out_degree', 'adar', 'preferential_attachment']

['jaccard', 'resource_allocation', 'shortest_path', 'follows_back','in_degree', 'out_degree', 'adar', 'preferential_attachment']

drop_list = ['jaccard', "preferential_attachment", 'follows_back', 'shortest_path', 'out_degree', 'adar']

X_train.drop(drop_list,inplace=True,axis=1)
X_test.drop(drop_list,inplace=True,axis=1)
X_dev.drop(drop_list,inplace=True,axis=1)

''' Step 3(Optional)
--------------------Normalization  --------------------------------------------
'''
scaler = StandardScaler() 
scaler.fit(X_train)  
X_train = scaler.transform(X_train) 
X_test = scaler.transform(X_test)

''' Step 4
--------------------Fit model ------------------------------------------------ 
'''
model = AdaBoostClassifier(n_estimators=100, random_state=1)
model.fit(X_train, Y_train)

''' Step 5(optional)
--------------------Evaluate-------------------------------------------
'''
y_pred = model.predict_proba(X_test)

dev_pred = model.predict_proba(X_dev)
dev_pred_score = model.predict(X_dev)
result_dev = np.array([pred[1] for pred in dev_pred])
accuracy = model.score(X_dev, Y_dev)
print("Accuracy: ", accuracy)

print("Other accuracy: ", metrics.accuracy_score(Y_dev, dev_pred_score))

fpr, tpr, thresholds = metrics.roc_curve(y_true = Y_dev, y_score = result_dev, pos_label=1)
print("AUC:", metrics.auc(fpr, tpr))

print("Other AUC:", roc_auc_score(y_true = Y_dev, y_score = result_dev))
plt.plot(fpr,tpr)

''' Step 6
--------------------Make  predictions-----------------------------------
'''
# the following are for testing purpose
results = [pred[1] for pred in y_pred]
results = pd.DataFrame(enumerate(results,1))
results.head()

''' Step 7
--------------------Export Results - Change your submission files--------------
'''
results.columns = ["Id","Predictions"]
results.head()
results.to_csv("./submissions/submission_AdaBoost_RA_ind_0.csv",index=False)