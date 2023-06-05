import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import os

from util import *

#load datasets
path = "Data_CSV\\3d_distances.csv"
X, y = load_datasets(path) 

print(type(X))
print(X.shape)
print(type(y))
print(y.shape)