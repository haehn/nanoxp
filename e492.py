import cv2
import numpy as np

def load_video(filename):

	video = cv2.VideoCapture(filename)
	length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
	width  = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
	fps    = video.get(cv2.CAP_PROP_FPS)

	return video, length, width, height, fps

def convert_to_bw(video, length, width, height):

	frames_color = np.zeros((length, height, width, 3), dtype=np.uint8)
	frames = np.zeros((length, height, width), dtype=np.uint8)
	for i in range(length):
	    
	    flag, frame = video.read()
	    frames_color[i] = frame
	    frames[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	return frames

def calculate_optical_flow(frames, startframe=1, endframe=160, N=5):

	# Parameters for lucas kanade optical flow
	lk_params = dict( winSize  = (15,15),
	                  maxLevel = 2,
	                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

	# params for ShiTomasi corner detection
	feature_params = dict( maxCorners = 100,
	                       qualityLevel = 0.3,
	                       minDistance = 7,
	                       blockSize = 7 )

	prev = frames[120] # TODO find better way
	p0 = cv2.goodFeaturesToTrack(prev, mask = None, **feature_params)

	# Create a mask image for drawing purposes
	mask = np.zeros_like(prev)
	# Create some random colors
	color = np.random.randint(0,255,(100,3))

	# holds all sliding windows
	all_sliding_windows = []

	for w in range(startframe, endframe):
	    for i in range(w,w+N):

	        prev = frames[i-1]
	        current = frames[i]


	        # calculate optical flow
	        p1, st, err = cv2.calcOpticalFlowPyrLK(prev, current, p0, None, **lk_params)

	        if p1 is None:
	            continue

	        # Select good points
	        good_new = p1[st==1]
	        good_old = p0[st==1]
	        # draw the tracks
	        for i,(new,old) in enumerate(zip(good_new, good_old)):
	            a,b = new.ravel()
	            c,d = old.ravel()
	            mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)

	        p0 = good_new.reshape(-1,1,2)
	        
	    # every 5 frames, we store our mask
	    # and clear it to zero
	    all_sliding_windows.append(mask)
	    mask = np.zeros_like(prev)

	return all_sliding_windows
