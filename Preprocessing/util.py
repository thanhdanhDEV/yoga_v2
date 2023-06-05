import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import os

# poses_list = ['pushups_up', 'pushups_down']

def create_datasets(poses_list, features_path, label_path, outcome_path):
    """create_datasets
    Load datasets from file csv. 

    :param poses_list: list of poses for classification, two poses
    :param features_path: path of data in project
    :param label_path: path of label in project
    
    :return None
    """

    # poses_list = ['pushups_up', 'pushups_down']
    # features_path = 'landmarks.csv'
    # label_path = 'labels.csv'
    df_x = pd.read_csv(features_path)
    df_y = pd.read_csv(label_path)

    data_pose_1 = df_x.loc[df_y.pose == poses_list[0]]
    data_pose_2 = df_x.loc[df_y.pose == poses_list[1]]

    label_pose_1 = df_y.loc[df_y.pose == poses_list[0]]
    label_pose_2 = df_y.loc[df_y.pose == poses_list[1]]

    X = pd.concat([data_pose_1, data_pose_2], ignore_index=True)
    y = pd.concat([label_pose_1, label_pose_2], ignore_index=True)

    y.replace(poses_list[0], "1", inplace= True)
    y.replace(poses_list[1], "0", inplace= True)

    #### X + y -> Data frames
    df = pd.concat([X, y], axis= 1)

    # ### Save to csv
    # os.makedirs('Data_CSV', exist_ok=True)  
    # outcome_path = 'landmarks.csv'
    df.to_csv(outcome_path, index=False)  
    return None

# path = 'Data_CSV\landmarks.csv'
def load_datasets(path):
    """load_datasets
    Load datasets from file csv. 

    :param path: relative path  of file csv
    
    :return X,  y: convert data frames, from reading file csv, to numpy (form of Data for scikit-learn==1.2)
    """
    df = pd.read_csv(path)
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]

    X = X.to_numpy()
    y = y.to_numpy()

    # print(type(X))
    # print(X.shape)
    # print(type(y))
    # print(y.shape)
    return X, y

