import pandas as pd 
import numpy as np

from util import *
import pickle

#load datasets
path = "Features_2_classes/3d_distances"
# path = "Features_2_classes/angles"

poses_list = ['lotus_pose', 'non_lotus_pose']

for j in range(len(poses_list)):
    path = path + "_" + poses_list[j]
path = path + ".csv"
# Load Data set

X, y = load_datasets(path) 

# Load Model

import time

filename = 'finalized_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))

start_time = time.time()
pred = loaded_model.predict(X[0:1])
print("--- %s seconds ---" % (time.time() - start_time))


#Metrics - Confusion Matrix
# from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score

# print("Confusion Matrix : \n",confusion_matrix(y, pred))
# print( "accuracy_score: ",accuracy_score(y, pred))
# print( "recall_score: ",recall_score(y, pred))
# print( "precision_score: ",precision_score(y, pred))