import cv2
import mediapipe as mp
import numpy as np
import math
from typing import List, Mapping, Optional, Tuple, Union


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
VISIBILITY_THRESHOLD = 0.5
PRESENCE_THRESHOLD = 0.5
# For static images:
IMAGE_FILES = []

STANDARD_ANGLE_POSE = []
# T_POSE = {'elbow_right' : [165, 175], 'elbow_left' : [165, 175], 'shoulder_right' : [85, 105], 'shoulder_left' : [85, 105], 'hip_right' :[170, 180], 'hip_left' : [170, 180], 'knee_right' : [170, 180], 'knee_left' : [170, 180]}
T_POSE = [[160, 175], [160, 175],[70, 105], [70, 105], [170, 180],[170, 180], [170, 180], [170, 180]]

T_POSE_1 = [[160, 180], [160, 180],[80, 105], [80, 105], [170, 180],[170, 180], [170, 180], [170, 180]]
T_POSE_2 = [[160, 180], [160, 180],[70, 90], [70, 90], [170, 180],[170, 180], [170, 180], [170, 180]]
T_POSE_3 = [[160, 180], [160, 180],[60, 80], [60, 80], [170, 180],[170, 180], [170, 180], [170, 180]]
T_POSE_4 = [[160, 180], [160, 180],[50, 70], [50, 70], [170, 180],[170, 180], [170, 180], [170, 180]]
T_POSE_LIST = [T_POSE_1, T_POSE_2, T_POSE_3, T_POSE_4]

FORWARD_BAND_1 = [[160, 180], [160, 180],[100, 170], [100, 170], [15, 30],[15, 30], [160, 180], [160, 180]]
FORWARD_BAND_2 = [[160, 180], [160, 180],[100, 170], [100, 170], [30, 40],[30, 40], [160, 180], [160, 180]]
FORWARD_BAND_3 = [[160, 180], [160, 180],[100, 170], [100, 170], [40, 55],[40, 55], [160, 180], [160, 180]]
FORWARD_BAND_4 = [[160, 180], [160, 180],[100, 170], [100, 170], [55, 100],[55, 100], [160, 180], [160, 180]]
FORWARD_BAND_LIST = [FORWARD_BAND_1, FORWARD_BAND_2, FORWARD_BAND_3, FORWARD_BAND_4]


# IMAGE_FILES.append('D:\Danh AI\MyProject\Yoga\Yogacheck\Image\yoga_easy.jpg')  ## Not FORWARD BEND - 0
# IMAGE_FILES.append('D:\Danh AI\MyProject\Yoga\Yogacheck\Image\Danh.png')  ## Not FORWARD BEND - 0
IMAGE_FILES.append('D:\Danh AI\MyProject\Yoga\Yogacheck\Image\i_t_pose.jpg')  ## Not FORWARD BEND - 0
# IMAGE_FILES.append('D:\Danh AI\MyProject\Yoga\Yogacheck\Image\i_forwardbend.jpg') ## FORWARD BEND - 500
# IMAGE_FILES.append('D:\Danh AI\MyProject\Yoga\Yogacheck\Image\i_forwardbend2.jpg') ## FORWARD BEND - 250

BG_COLOR = (192, 192, 192)  # gray
LIST_POST = ["NOSE", "LEFT_EYE_INNER", "LEFT_EYE", "LEFT_EYE_OUTER", "RIGHT_EYE_INNER", "RIGHT_EYE", "RIGHT_EYE_OUTER",
             "LEFT_EAR", "RIGHT_EAR", "MOUTH_LEFT", "MOUTH_RIGHT", "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW",
             "RIGHT_ELBOW", "LEFT_WRIST", "RIGHT_WRIST", "LEFT_PINKY", "RIGHT_PINKY", "LEFT_INDEX", "RIGHT_INDEX",
             "LEFT_THUMB", "RIGHT_THUMB", "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE",
             "RIGHT_ANKLE", "LEFT_HEEL", "RIGHT_HEEL", "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX", ]

def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.rad2deg(radians)
    
    if abs(angle) > 180.0:
        angle = 360-abs(angle)
    return abs(angle) 

def is_pose_in_POSTLIST(list_angle, POSE):
    is_pose_in = True
    for i in range(len(list_angle)):
        if list_angle[i] > POSE[i][0] and list_angle[i] < POSE[i][1]:
            continue
        # print(i) #determine number incorrect coordinates
        is_pose_in = False
        break
    return is_pose_in

def pose_detection(list_angle, POSE_LIST):
    mask_list = [500, 375, 250, 125]
    # is_yoga_pose = "FORWARD BEND"
    is_yoga_pose = "T POSE"
    mask = 125
    color = (0, 255, 0)
    # print(POSE_LIST[0][0]) #debug - test
    # print(list_angle[0])
    for i in range(len(POSE_LIST)):
        if is_pose_in_POSTLIST(list_angle, POSE_LIST[i]): 
            # is_yoga_pose = "FORWARD BEND"
            is_yoga_pose = "T POSE"
            mask = mask_list[i]
            color = (0, 255, 0)
            break
        else:
            # is_yoga_pose = "Not FORWARD BEND"
            is_yoga_pose = "Not T POSE"
    if is_yoga_pose == "Not T POSE":
        color = (0, 0, 255)
        mask = 0
    return is_yoga_pose, color, mask

def normalized_to_pixel_coordinates(
        normalized_x: float, normalized_y: float, image_width: int,
        image_height: int) -> Union[None, Tuple[int, int]]:
    # Checks if the float value is between 0 and 1.
    def is_valid_normalized_value(value: float) -> bool:
        return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                          math.isclose(1, value))

    if not (is_valid_normalized_value(normalized_x) and
            is_valid_normalized_value(normalized_y)):
        return None
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    return (x_px, y_px)