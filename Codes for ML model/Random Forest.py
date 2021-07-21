''' Step 0
--------------------Reset parameter & import required package  ----------------------
'''
%reset -f

import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np
import csv
import matplotlib.pyplot as plt
import os,pathlib
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.metrics import roc_curve, auc


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

drop_list = []

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

''' Step 4(Optional)
--------------------Feature Selection ------------------------------------------------
'''
# Feature Importance
# display the relative importance of each attribute
feature_importance = model.feature_importances_
print(model.feature_importances_)

''' Step 5
--------------------Fit model ------------------------------------------------
'''
model = RandomForestClassifier(n_estimators=50000, criterion="gini")
model = ExtraTreesClassifier()
model.fit(X_train, Y_train)

''' Step 6(optional)
--------------------Evaluate-------------------------------------------
'''
# Accuracy Score 
accuracy = model.score(X_dev, Y_dev)
print("Accuracy: ", accuracy)

# Compute the ROC curves
y_pred_dev = model.predict(X_dev)
fpr_1, tpr_1, thresholds_1 = roc_curve(Y_dev, y_pred_dev, pos_label=1)
# Compute the AUC
auc_score = auc(fpr_1, tpr_1)

# Make standard ROC plot
%matplotlib inline
plt.figure()
plt.plot(fpr_1, tpr_1, color='b', label='Model 1 (AUC = %0.2f)' % auc_score)
plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC curves')
plt.legend(loc='lower right')
plt.show()

''' Step 7
--------------------Make  predictions-----------------------------------
''' 
y_pred = model.predict_proba(X_test)
            
''' Step 8
--------------------Export Results - Change your submission files--------------
'''
results.columns = ["Id","Predictions"]
results.head()
results.to_csv("./Result/submission_RF_all.csv",index=False)        

        
        
    
        
