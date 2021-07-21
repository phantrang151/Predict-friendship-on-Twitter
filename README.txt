There are 4 code folders corresponding to 4 types of tasks:
Task 1. Data Extraction
Task 2. Exploratory Analysis
Task 3. Feature Creation
Task 4. Model Selection

Details for each task is provided below:

Task 1. Data Extraction: extract only a portion from the original data.
The .py files were marked with suffix: Step1, Step 2, etc. and should be run this order.
Input: 'train.txt' and test-public.txt'
Output:
- TOTAL_EDGES: all the edges from the file 'train.txt'
- TOTAL_NODES: all the nodes from the file 'train.txt'
_ TEST_EDGES: all the edges from the file 'public-test.txt'
_ TEST_NODES: all the nodes from the file 'public-test.txt'
- train_pos_x_100K: 100K positive edges for training
- train_neg_x_100K: 100K negative edges for training
- train_x_200K: 200K edges for training
- train_pos_y_100K: 100K labels for training, all values are 1
- train_neg_y_100K: 100K labels for training, all values are 0
- train_y_200K: 200K labels for training
- dev_pos_x_25K: 25K positive edges for dev
- dev_neg_x_25K: 25K negative edges for dev
- dev_x_50K: 50K edges for dev
- dev_pos_y_25K: 25K labels for dev, all values are 1
- dev_neg_y_25K: 25K labels for dev, all values are 0
- dev_y_50K: 50K labels for dev

Task 2. Exploratory Analysis: to get some understanding about the graph and how edges are distributed
Please run the file: 'Exploratory Analysis.py'.
Input: 'TOTAL_EDGES.csv' and 'TEST_NODES.csv'
Output: printed lines displaying graph information

Task 3. Feature Creation: to generate potential features used for training.
Please run 3 files:
- CreateFeatures_TrainData.py
_ CreateFeatures_TestData
_ CreateFeatures_DevData

Input: 'TOTAL_EDGES.csv', 'TEST_EDGES.csv','dev_x_50K.csv', 'train_x_200K.csv's
Output: 
- train_x_ML_200K.pkl: for training
- test_x_ML_200K.pkl : for testing
- dev_x_ML_50K.pkl   : for dev

Alternative method:
In the Reference folder, we also include a list of .py files that are used to generate independent features, 1 feature at a time.
This method helps us to generate all features for training with limited computing resources.
Please run all of the files with prefix: 'create_feature', and run 'combine_all_features.py' at the end
to combine all inputs into 1 pickle file.

Task 4. Model Selection: multiple models are used for prediction.
All the .py files in this folder can be run independently, each file represent 1 model.
Template code for the following models: Random Forest, Logistic Regression, Articial Neural Network (ANN), Stacking Model with XGBClassifier as meta-model.
There is also a .py file that is used for feature exploration and feature selection exercise.
Input: all generated files in Task1 and Task3
Output: csv file containing link prediction, ready for submission 

Additional Note: in each file, there is a pathlib.Path() section, please enter the folder where the extracted data should be placed.
The syntax is: path = pathlib.Path([place of the generated files])