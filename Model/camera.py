# import the opencv library
import cv2
import numpy as np
# define a video capture object
vid = cv2.VideoCapture(0)

def make_720p():
    vid.set(3, 1280)
    vid.set(4, 720)

# make_720p()
# ret, frame = vid.read()
# img_v = cv2.flip(frame, 1)
# print(type(img_v))
# print(img_v.shape)

make_720p()
count = 1
while(True):
      
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    img_v = cv2.flip(frame, 1)
    # print("image is : ", np.array(img_v).shape())
    # Display the resulting frame
    cv2.putText(img=img_v, text='Lotus Pose', org=(50, 50), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1.5,
 color=(0, 0, 255),thickness=2)
    nb = 'Count: ' + str(count)
    
    cv2.putText(img=img_v, text='Count: ' + str(count), org=(500, 50), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1.5, color=(255, 0, 0),thickness=2)
    count = count  + 1
    cv2.imshow('frame', img_v)
    

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()