''' Step 0
--------------------Reset parameter & import required package  ----------------------
'''
%reset -f

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from vecstack import stacking
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import os,pathlib
import numpy as np


''' Step 1
--------------------Change configurations & file names  ----------------------
'''
path = pathlib.Path(r"C:\Users\saisu\Desktop\Module3\Machine Learning\Assignment")
os.chdir(path)


X_train = pd.read_pickle('./pkl/train_x_ML_200K.pkl')
Y_train = pd.read_csv('./Data Final/train_y_200K.csv',  header=None)
Y_train = np.array(Y_train[0]) # convert to array, using only the first column

X_dev = pd.read_pickle('./pkl/dev_x_ML_50K.pkl')
Y_dev =  pd.read_csv('./Data Final/dev_y_50K.csv',  header=None)
Y_dev = np.array(Y_dev[0])

X_test = pd.read_pickle('./pkl/test_x_ML_2K.pkl')


''' Step 2(Optional)
--------------------Drop the features  ------------------------------------
'''
# FULL LIST: 
# ['jaccard', 'resource_allocation', 'shortest_path', 'follows_back','in_degree', 'out_degree', 'adar', 'preferential_attachment']

drop_list = ['jaccard', 'follows_back','in_degree', 'out_degree', 'adar', 'preferential_attachment']

X_train.drop(drop_list,inplace=True,axis=1)
X_test.drop(drop_list,inplace=True,axis=1)
X_dev.drop(drop_list,inplace=True,axis=1)

''' Step 3
--------------------Setup model ------------------------------------------------
'''
models = [
    RandomForestClassifier(n_estimators=1000, criterion="gini"),
    LogisticRegression(solver="lbfgs", max_iter=10000, multi_class="auto"),
    MLPClassifier(hidden_layer_sizes=(10,10), activation='relu',max_iter=10000),
    LogisticRegression(solver="lbfgs", max_iter=10000, multi_class="auto"),
    MLPClassifier(hidden_layer_sizes=(10,10), activation='logistic',max_iter=10000)
    ]


''' Step 4
--------------------Tuning parameter------------------------------------------------
'''
'''
models: the first level models we defined earlier
X_train, y_train, X_test: our data
regression: Boolean indicating whether we want to use the function for regression. In our case set to False since this is a classification
mode: using the earlier describe out-of-fold during cross-validation
needs_proba: Boolean indicating whether you need the probabilities of class labels
save_dir: save the result to directory Boolean
metric: what evaluation metric to use (we imported the accuracy_score in the beginning)
n_folds: how many folds to use for cross-validation
stratified: whether to use stratified cross-validation
shuffle: whether to shuffle the data
random_state: setting a random state for reproducibility
verbose: 2 here refers to printing all info
'''

S_train, S_test = stacking(models,                   
                           X_train, Y_train, X_test,
                           regression=False,      
                           mode='oof_pred_bag',        
                           needs_proba=False,         
                           save_dir=None,             
                           metric=roc_auc_score,     
                           n_folds=5,                  
                           stratified=True,            
                           shuffle=True,              
                           random_state=0,             
                           verbose=2)

''' Step 5
--------------------Fit model------------------------------------------------
Boosting Classifier
'''
model = XGBClassifier(random_state=0, n_jobs=-1, learning_rate=0.1, 
                      n_estimators=300, max_depth=4, booster = 'gbtree')
    
model = model.fit(S_train, Y_train)

''' Step 6 (Optional)
--------------------Evaluate-----------------------------------------------
SHOULD CHANGE "X_test" to "X_dev" AND RUN SEPERATE WITH "X_test".
'''
y_pred = model.predict(S_test)
result = model.predict_proba(S_test)
#print('Final prediction score: [%.8f]' % roc_auc_score(Y_dev, y_pred))

''' Step 7
--------------------Make  predictions-----------------------------------
''' 
y_pred = model.predict(S_test)


''' Step 8
--------------------Export Results - Change your submission files--------------
'''
results = [pred[1] for pred in result]

df_result = pd.DataFrame(enumerate(results,1))
df_result.columns=["Id","Predictions"]

df_result.to_csv("./Result/Stacking_RA_and_SP_v1.csv",index=False)
