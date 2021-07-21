''' Step 0
--------------------Reset parameter & import required package  ----------------------
'''
%reset -f

import tensorflow as tf
import pandas as pd
import os,pathlib
import pickle
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


''' Step 3(Optional)
--------------------Normalization  --------------------------------------------
'''
X_train = tf.keras.utils.normalize(X_train, axis = 1)
X_test = tf.keras.utils.normalize(X_test, axis = 1)
X_dev = tf.keras.utils.normalize(X_dev, axis = 1)

''' Step 4
--------------------Setup model ------------------------------------------------
'''
model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Dense(64, activation='relu', input_dim = 8))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(2, activation='softmax'))

sgd = tf.keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

''' Step 5
--------------------Train model ------------------------------------------------
'''
model.fit(X_train, y_train,
          epochs=5,
          batch_size=128)


''' Step 6
--------------------Evaluate-----------------------------------------------
'''
dev_loss, dev_acc = model.evaluate(X_dev,y_dev)
result_dev = model.predict_proba(X_dev.values)
print("test accurary:", dev_acc)
auc, update_op = tf.metrics.auc(labels = y_dev,predictions = result_dev[:,1])
with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    print("tf auc: {}".format(sess.run([auc, update_op])))
    
''' Step 7
--------------------Make  predictions-----------------------------------
'''  
result = model.predict_proba(X_test.values)

''' Step 8
--------------------Export Results - Change your submission files--------------
'''

results = [pred[1] for pred in result]

df_result = pd.DataFrame(enumerate(results,1))
df_result.columns=["Id","Predictions"]

df_result.to_csv("./Result/Stacking_RA_and_SP_v1.csv",index=False)