# Importing all necessary libraries
import cv2
import os

# Read the video from specified path

# video_path = 'path of video at here'
video_path = 'Video_demo/non_lotus_pose_7_4_2023.mp4'
cam = cv2.VideoCapture(video_path)
SAVING_FRAMES_PER_SECOND = 30


try:
	# creating a folder named data
	if not os.path.exists('images_non_lotus_pose'):
		os.makedirs('images_non_lotus_pose')

# if not created then raise error
except OSError:
	print ('Error: Creating directory of data')

# frame
currentframe = 0
count = 1
while(True):
	
	# reading from frame
	ret,frame = cam.read()

	if ret:
		if (currentframe % 5) == 0:
			# if video is still left continue creating images
			name = './images_non_lotus_pose/lotus_' + str(count) + '.jpg'
			print ('Creating...' + name)

		    # writing the extracted images
			cv2.imwrite(name, frame)

		    # increasing counter so that it will
		    # show how many frames are created
			count += 1
		currentframe += 1
	else:
		break

# Release all space and windows once done
cam.release()
cv2.destroyAllWindows()

