#%reset -f
"""
--------------------------  4 steps -------------------------------------------
"""
import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np
import csv
import os,pathlib
from xgboost import XGBClassifier

''' Step 1
--------------------Change configurations & file names  ----------------------
'''
path = pathlib.Path(r"C:\Users\Trang\Master of Business Analytics\Module 3\Machine Learning\Project\Version4 - Random 100K - 24M Edges")
os.chdir(path)

X_train = pd.read_pickle('features_resource_allocation.pkl')
X_test = pd.read_pickle('test_features_resource_allocation.pkl')
Y_train = pd.read_csv('train_y_200K.csv',  header=None)
Y_train = np.array(Y_train[0]) # convert to array, using only the first column


''' Step 2 - Optional - Drop the features
'''

#X_train.drop(["source","target","jaccard","shortest_path","follows_back"],inplace=True,axis=1)
#X_test.drop(["jaccard","shortest_path","follows_back"],inplace=True,axis=1)

X_test
X_test["resource_allocation"].min()


''' Step 3
--------------------Fit model ------------------------------------------------
'''
#model = LogisticRegression(solver="lbfgs", max_iter=10000, multi_class="auto")
model = XGBClassifier()

model.fit(X_train, Y_train)
y_pred = model.predict_proba(X_test)
y_pred

# the followings are for testing purpose
results = [pred[1] for pred in y_pred]
results

(np.array(results)).mean()
  
        
#results.head()  
results = pd.DataFrame(enumerate(results,1))  
        
''' Step 4
--------------------Export Results - Change your submission files--------------
'''
results.columns = ["Id","Predictions"]
results.head()   
        
results.to_csv("submission_XGBoost.csv",index=False)        

        
        
        
        
        
        
        
        
        
        