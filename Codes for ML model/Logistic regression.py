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
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
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

drop_list = ['in_degree', 'out_degree', 'adar', 'preferential_attachment']

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
model = LogisticRegression(solver="lbfgs", max_iter=10000, multi_class="auto")
model.fit(X_train, Y_train)

''' Step 5(optional)
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
plt.figure()
plt.plot(fpr_1, tpr_1, color='b', label='Model 1 (AUC = %0.4f)' % auc_score)
plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC curves')
plt.legend(loc='lower right')
plt.show()

''' Step 6
--------------------Make  predictions-----------------------------------
''' 
y_pred = model.predict_proba(X_test)
        
''' Step 7
--------------------Export Results - Change your submission files--------------
'''
results.columns = ["Id","Predictions"]
results.head()
results.to_csv("./Result/submission_LR_SP.csv",index=False)        
        
        
        
        