import numpy as np
import pandas as pd 
import os


"""

# list of image path
out_list = list(map(str, list(range(1,9))))
image_list = [item + ".jpg" for item in out_list]
# print(image_list)

"""
# list of label
labels_pose_1 = ["squat_up" for i in range(327)] # pose 1 - squat up
labels_pose_2 = ["squat_down" for i in range(268)] # pose 2 - squat down
labels = labels_pose_1 + labels_pose_2 # labels

labels_series = pd.Series(name= "pose", data=labels)
labels_series = pd.DataFrame(labels_series)
print(labels_series.pose.value_counts())

### Save to csv
# os.makedirs('Data_CSV', exist_ok=True)
outcome_path = 'labels.csv'
labels_series.to_csv(outcome_path, index=False)


# rename all file
# dir_path = "Extract_image_test/"
# all_files_and_folders = os.listdir(dir_path)
# k = len(os.listdir(dir_path))
# print(k)
# for i in range(len(all_files_and_folders)):
#     os.rename("Extract_image_pose2/" + all_files_and_folders[i], "Extract_image_pose2/" + str(i+1)+ ".jpg")