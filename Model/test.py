import pandas as pd 
import numpy as np

from util import *
from yoga_model import Yoga_Model
import pickle
import time

# path_data = 'Features_2_classes/3d_distances_lotus_pose_non_lotus_pose.csv'
# X, y = load_datasets(path_data) 

# filename = 'finalized_model.sav'
# loaded_model = pickle.load(open(filename, 'rb'))

# mod = Yoga_Model(loaded_model)

# for i in range(10):
#     start_time = time.time()
#     pred = mod.predict(X[i+370].reshape((1,-1)))
#     print("Label is %d" %pred)
#     print("--- %s seconds ---" % (time.time() - start_time))

for i in range(10):
    print('Count: ' + str(i))