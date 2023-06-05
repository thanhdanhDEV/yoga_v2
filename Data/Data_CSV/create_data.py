import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import os

from util import *

poses_list = ['pushups_up', 'pushups_down']
features_path = '3d_distances.csv'
label_path = 'labels.csv'
outcome_path = '3d_distances.csv'
create_datasets(poses_list= poses_list, features_path= features_path, label_path= label_path, outcome_path= outcome_path)

