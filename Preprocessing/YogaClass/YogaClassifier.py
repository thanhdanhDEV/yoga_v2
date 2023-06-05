import math
import pandas as pd 
import numpy as np

# Data Class
class DataIn():
    """
        input: path_landmarks, path_labels
        output: df_landmarks, df_labels
    """
    def __init__(self, path_landmarks, path_labels):
        self.path_landmarks = path_landmarks
        self.path_labels = path_labels
        self.df_landmarks, self.df_labels = self.__read_data()

    def __read_data(self):
        df_landmarks = pd.read_csv(self.path_landmarks)
        df_labels = pd.read_csv(self.path_labels)
        return df_landmarks, df_labels

class DataOut():
    """
        input: path_landmarks, path_labels
        output: df_landmarks, df_labels
    """
    def __init__(self):
        self.df_distance = pd.DataFrame()
        self.df_angles = pd.DataFrame()
        self.df_labels = pd.DataFrame()



class PreProcess:

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
    
    name_angles = ['right_elbow_right_shoulder_right_hip',
            'left_elbow_left_shoulder_left_hip',
            'right_knee_mid_hip_left_knee',
            'right_hip_right_knee_right_ankle',
            'left_hip_left_knee_left_ankle',
            'right_wrist_right_elbow_right_shoulder',
            'left_wrist_left_elbow_left_shoulder']
    
    def __init__(self, dataInput,  dataOutput):
        self.dataInput = dataInput
        self.dataOutput = dataOutput

    def cal_distances(self, point_1, point_2):
        # [x1_coords, y1_coords, z1_coords] = [ -5.889507, -57.63752, -45.01975]
        # [x2_coords, y2_coords, z2_coords] = [ -4.656085, -62.832863, -44.571823]
        p1 = np.array([point_1[0], point_1[1], point_1[2]])
        p2 = np.array([point_2[0], point_2[1], point_2[2]])

        squared_dist = np.sum((p1-p2)**2, axis=0)
        dist = np.sqrt(squared_dist)
        return dist
    
    def create_dist_Series(self,dist_between_string):
        ### split string
        index = dist_between_string.find("_", int(len(dist_between_string)/2 - 2), -1) ### find underscore in mid string
        A_point = dist_between_string[:index]
        B_point = dist_between_string[index+1:]

        # Define the function
        def norm_2(point_1, point_2):
            A = [point_1[0], point_1[1], point_1[2]]
            B = [point_2[0], point_2[1], point_2[2]]
            dist = self.cal_distances(point_1= A, point_2= B)
            return dist

        # Apply the function to the DataFrame
        out_distances = self.dataInput.df_landmarks.apply(lambda row: norm_2(row[['x_'+A_point, 'y_'+A_point, 'z_'+A_point]], row[['x_'+B_point, 'y_'+B_point, 'z_'+B_point]]), axis=1)
        return out_distances

    def create_avg_dist(self, avg_between_string):
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
        out_distances = self.dataOutput.df_distance.apply(lambda row: avg_(row), axis= 1)
        return out_distances

    def generate_dist(self, name_distances = name_distances):
        for i in  range(len(name_distances)):
            index = name_distances[i].find("avg_")
            if index == -1:
                self.dataOutput.df_distance[name_distances[i]] = self.create_dist_Series(name_distances[i])
            else:
                self.dataOutput.df_distance[name_distances[i]] = self.create_avg_dist(name_distances[i])
        self.dataOutput.df_distance = self.dataOutput.df_distance.astype(np.float32)
        self.dataOutput.df_labels = self.dataInput.df_labels
    
    def cal_angle(self, point_1, point_2,  point_3):
        # [x1_coords, y1_coords, z1_coords] = [ -5.889507, -57.63752, -45.01975]
        # [x2_coords, y2_coords, z2_coords] = [ -4.656085, -62.832863, -44.571823]
        p1 = np.array([point_1[0], point_1[1], point_1[2]])
        p2 = np.array([point_2[0], point_2[1], point_2[2]])
        p3 = np.array([point_3[0], point_3[1], point_3[2]])

        v1 = p1 - p2
        v2 = p3 - p2

        def norm_2(vector):
            v_norm2 = np.sum((vector)**2, axis=0)
            v_norm2 = np.sqrt(v_norm2)
            return v_norm2
        
        v1_norm = norm_2(v1)
        v2_norm = norm_2(v2)

        angle = math.acos((np.sum(v1*v2))/(v1_norm*v2_norm))
        angle = angle * 180 / math.pi
        return angle
    
    def create_dist_Series_angle(self, three_points_string):
        ### split string
        three_points_string = three_points_string.split("_")

        list_index = []
        for i, item in enumerate(three_points_string):
            if item in ["right", "left", "mid"]:
                list_index.append(i)

        A_point = "_".join(three_points_string[list_index[0]:list_index[1]])
        B_point = "_".join(three_points_string[list_index[1]:list_index[2]])
        C_point = "_".join(three_points_string[list_index[2]:])

        # Define the function
        def multiply(point_1, point_2, point_3):
            A = [point_1[0], point_1[1], point_1[2]]
            B = [point_2[0], point_2[1], point_2[2]]
            C = [point_3[0], point_3[1], point_3[2]]
            dist = self.cal_angle(point_1= A, point_2= B, point_3= C)
            return dist

        # Apply the function to the DataFrame
        out_angle = self.dataInput.df_landmarks.apply(lambda row: multiply(row[['x_'+A_point, 'y_'+A_point, 'z_'+A_point]], row[['x_'+B_point, 'y_'+B_point, 'z_'+B_point]], row[['x_'+C_point, 'y_'+C_point, 'z_'+C_point]]), axis=1)
        return out_angle

    def generate_angle(self, name_angles = name_angles):
        # Create mid hip - coordination
        self.dataInput.df_landmarks["x_mid_hip"] = (self.dataInput.df_landmarks.x_right_hip + self.dataInput.df_landmarks.x_left_hip) / 2
        self.dataInput.df_landmarks["y_mid_hip"] = (self.dataInput.df_landmarks.y_right_hip + self.dataInput.df_landmarks.y_left_hip) / 2
        self.dataInput.df_landmarks["z_mid_hip"] = (self.dataInput.df_landmarks.z_right_hip + self.dataInput.df_landmarks.z_left_hip) / 2

        for i in range(len(name_angles)):
            self.dataOutput.df_angles[name_angles[i]] = self.create_dist_Series_angle(name_angles[i])
        
        self.dataOutput.df_angles = self.dataOutput.df_angles.astype(np.float16)

class YogaClassifier:
    def __init__(self, name, age):
        self.name = name
        self.age = age
        self.__age = age

    def info(self):
        print(self.name)

    def update_age(self, new_age):
        self.__age = new_age
        self.age = self.__age

if __name__ == "__main__":
    # YogaPush = YogaClassifier("Push Classifier", 10)
    # YogaPush.info()
    # print(YogaPush.age)

    dataIn = DataIn(path_landmarks= "../Data_Real/landmarks.csv", path_labels= "../Data_Real/labels.csv")
    dataOut = DataOut()
    SquatPreprocessing = PreProcess(dataInput= dataIn, dataOutput= dataOut)

    SquatPreprocessing.generate_dist()
    SquatPreprocessing.generate_angle()

    Output_Pre = SquatPreprocessing.dataOutput

    print(Output_Pre.df_distance.info())
    print(Output_Pre.df_angles.info())
    print(Output_Pre.df_labels.info())
    pass