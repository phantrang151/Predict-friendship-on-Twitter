#%reset -f
"""
--------------------------  4 steps -------------------------------------------
"""
import pandas as pd
from xgboost import XGBClassifier
import numpy as np
from sklearn.feature_selection import SelectFromModel
import os,pathlib
from sklearn.metrics import accuracy_score
from matplotlib import pyplot
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
import numpy

''' Step 1
--------------------Change configurations & file names  ----------------------
'''
path = pathlib.Path(r"C:\Users\Trang\Master of Business Analytics\Module 3\Machine Learning\Project\Version2 - Random 500K - return 0")
os.chdir(path)

X_train = pd.read_pickle('features_resource_allocation.pkl')
Y_train = pd.read_csv('train_y.csv',  header=None)
Y_train = np.array(Y_train[0]) # convert to array, using only the first column

X_dev = pd.read_pickle('dev_x_ML_50K.pkl')
Y_dev = pd.read_csv('dev.csv',  header=None)
Y_dev = np.array(Y_dev[0]) # convert to array, using only the first column

X_test = pd.read_pickle('test_x_ML_100K.pkl')
Y_test = pd.read_pickle('test_y_ML_100K.pkl',  header=None)
Y_dev = np.array(Y_test[0])


''' Step 2 - Optional - Drop the features
'''
#X_train.drop(["source","target","jaccard","shortest_path","follows_back"],inplace=True,axis=1)
#X_test.drop(["jaccard","shortest_path","follows_back"],inplace=True,axis=1)

#X_test.head()
#X_test["resource_allocation"].min()


''' Step 3
--------------------Fit model ------------------------------------------------
'''
model = XGBClassifier()
model.fit(X_train, Y_train)
from numpy import sort

''' Step 4
-------------------- Feature Selection ----------------------------------------
'''

y_pred = model.predict(X_dev)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(Y_dev, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
# Fit model using each importance as a threshold
thresholds = sort(model.feature_importances_)
for thresh in thresholds:
	# select features using threshold
	selection = SelectFromModel(model, threshold=thresh, prefit=True)
	select_X_train = selection.transform(X_train)
	# train model
	selection_model = XGBClassifier()
	selection_model.fit(select_X_train, Y_train)
	# eval model
	select_X_dev = selection.transform(X_dev)
	y_pred = selection_model.predict(select_X_dev)
    
	predictions = [round(value) for value in y_pred]
	accuracy = accuracy_score(Y_test, predictions)
	print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], accuracy*100.0))


''' Step 5
-------------------- Choose the number of trees and tree sizes ----------------
'''

model1 = XGBClassifier()
n_estimators1 = [50, 100, 150, 200]
max_depth1 = [2, 4, 6, 8]

param_grid = dict(max_depth=max_depth1, n_estimators=n_estimators1)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
grid_search = GridSearchCV(model1, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold, verbose=1)
grid_result = grid_search.fit(X_train, Y_train)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
	print("%f (%f) with: %r" % (mean, stdev, param))
# plot results
scores = numpy.array(means).reshape(len(max_depth1), len(n_estimators1))
for i, value in enumerate(max_depth1):
    pyplot.plot(n_estimators1, scores[i], label='depth: ' + str(value))
pyplot.legend()
pyplot.xlabel('n_estimators')
pyplot.ylabel('Log Loss')
pyplot.show()
        
''' Step 6
---------------Finished choosing features and tree size and kfold--------------
'''
# adjust the features that we want by dropping the columns
# --------------------------------------------------------
final_model = XGBClassifier(n_estimators=5000, max_depth=4)
final_model.fit(X_train, Y_train)  #to key in the parameters here
y_pred = final_model.predict_proba(X_test)
results = [pred[1] for pred in y_pred]
results = pd.DataFrame(enumerate(results,1))  
results.columns = ["Id","Predictions"]
results.head()   
        
results.to_csv("submission_XGBoost.csv",index=False)        

'''  KFold using DEV data
n_folds = 10
kfold = KFold(n_folds, True, 1)
# cross validation estimation of performance
scores, members = list(), list()
for train_ix, test_ix in kfold.split(X):
	# select samples
	trainX, trainy = X[train_ix], y[train_ix]
	testX, testy = X[test_ix], y[test_ix]
	# evaluate model
	model, test_acc = evaluate_model(trainX, trainy, testX, testy)
	print('>%.3f' % test_acc)
	scores.append(test_acc)
	members.append(model)

def evaluate_model(trainX, trainy, testX, testy):
	# encode targets
	trainy_enc = to_categorical(trainy)
	testy_enc = to_categorical(testy)
	# define model
	model = Sequential()
	model.add(Dense(50, input_dim=2, activation='relu'))
	model.add(Dense(3, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	# fit model
	model.fit(trainX, trainy_enc, epochs=50, verbose=0)
	# evaluate the model
	_, test_acc = model.evaluate(testX, testy_enc, verbose=0)
	return model, test_acc


# summarize expected performance
print('Estimated Accuracy %.3f (%.3f)' % (mean(scores), std(scores)))
'''        
        
        
        
        
        
        
        
        
        