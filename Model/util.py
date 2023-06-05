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

def convert_2_3d_distance(df_landmarks):
    def cal_distances(point_1, point_2):
        # [x1_coords, y1_coords, z1_coords] = [ -5.889507, -57.63752, -45.01975]
        # [x2_coords, y2_coords, z2_coords] = [ -4.656085, -62.832863, -44.571823]
        p1 = np.array([point_1[0], point_1[1], point_1[2]])
        p2 = np.array([point_2[0], point_2[1], point_2[2]])

        squared_dist = np.sum((p1-p2)**2, axis=0)
        dist = np.sqrt(squared_dist)
        return dist

    ### get list of columns : df_distances

    def create_dist_Series(dist_between_string):
        ### split string
        index = dist_between_string.find("_", int(len(dist_between_string)/2 - 2), -1) ### find underscore in mid string
        A_point = dist_between_string[:index]
        B_point = dist_between_string[index+1:]

        # Define the function
        def norm_2(point_1, point_2):
            A = [point_1[0], point_1[1], point_1[2]]
            B = [point_2[0], point_2[1], point_2[2]]
            dist = cal_distances(point_1= A, point_2= B)
            return dist

        # Apply the function to the DataFrame
        out_distances = df_landmarks.apply(lambda row: norm_2(row[['x_'+A_point, 'y_'+A_point, 'z_'+A_point]], row[['x_'+B_point, 'y_'+B_point, 'z_'+B_point]]), axis=1)
        return out_distances

    def create_avg_dist(avg_between_string):
        index = avg_between_string.find("avg_") ### find avg_ in mid string
        A_point = avg_between_string[:index-1]
        other = avg_between_string[index+4:]

        sub_str = other
        index = sub_str.find("_", int(len(sub_str)/2 - 2), -1) ### find underscore in mid string
        B_point = sub_str[:index]
        C_point = sub_str[index+1:]
        def avg_(row):
            return (row[A_point+ '_' + B_point] + row[A_point+ '_' + C_point]) /2

        # Apply the function to the DataFrame
        out_distances = df_out_distances.apply(lambda row: avg_(row), axis= 1)
        return out_distances

    # Importing Pandas to create DataFrame
    import pandas as pd
    
    # Creating Empty DataFrame and Storing it in variable df
    df_out_distances = pd.DataFrame()

    # name_distances = list(df_distances.columns)

    name_distances = ['left_shoulder_left_wrist',
                    'right_shoulder_right_wrist',
                    'left_hip_left_ankle',
                    'right_hip_right_ankle',
                    'left_hip_left_wrist',
                    'right_hip_right_wrist',
                    'left_shoulder_left_ankle',
                    'right_shoulder_right_ankle',
                    'left_hip_right_wrist',
                    'right_hip_left_wrist',
                    'left_elbow_right_elbow',
                    'left_knee_right_knee',
                    'left_wrist_right_wrist',
                    'left_ankle_right_ankle',
                    'left_hip_avg_left_wrist_left_ankle',
                    'right_hip_avg_right_wrist_right_ankle']

    for i in  range(len(name_distances)):
        index = name_distances[i].find("avg_")
        if index == -1:
            df_out_distances[name_distances[i]] = create_dist_Series(name_distances[i])
        else:
            df_out_distances[name_distances[i]] = create_avg_dist(name_distances[i])
    features_3d_distances = df_out_distances.iloc[0].to_numpy().reshape((1,-1))
    return features_3d_distances