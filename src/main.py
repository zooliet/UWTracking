# import the necessary packages

from __future__ import print_function
import sys
PY3 = sys.version_info[0] == 3
if PY3:
	xrange = range


import cv2
import numpy as np
import imutils
from imutils.video import WebcamVideoStream
from imutils.video import FPS
import hashlib

from trackers.color_tracker import ColorTracker
from trackers.kcf_tracker import KCFTracker

from motor import Motor


import math
from threading import Timer
import time
import datetime

from utils import common
from utils import util

# from numpy import isnan
# import redis

# import json
# from common import draw_str
# import time
# import util

# construct the argument parse and parse the arguments
import argparse

ap = argparse.ArgumentParser()
# ap.add_argument("--camera", type=int, default=0, help = "camera number")
ap.add_argument("--camera", type=int, default=0, help = "camera number")
ap.add_argument("-p", "--path", help = "path to video file")
ap.add_argument("-n", "--num-frames", type=int, default=1000000, help="# of frames to loop over")
ap.add_argument("-d", "--display", action="store_true", help="Show display")
ap.add_argument("-f", "--fifo", action="store_true", help="Enable FIFO for ffmpeg")
ap.add_argument("-s", "--serial", help = "path to serial device")
ap.add_argument("-z", "--zoom", help = "path to zoom control port")

ap.add_argument("--color", action="store_true", help="Enable color tracking")
ap.add_argument("--kcf", action="store_true", help="Enable KCF tracking")

# ap.add_argument('--motion', dest='motion', action='store_true', help='Enable motion tracking')
# ap.add_argument('--cmt', dest='cmt', action='store_true', help='Enable CMT tracking')
# ap.add_argument('--optical', dest='optical', action='store_true', help='Enable optical flow tracking')
# ap.add_argument('--with-scale', dest='estimate_scale', action='store_true', help='Enable scale estimation')
# ap.add_argument('--with-rotation', dest='estimate_rotation', action='store_true', help='Enable rotation estimation')
args = vars(ap.parse_args())
# print(args)


WIDTH       = 640  # 640x360, 1024x576, 1280x720, 1920x1080
HEIGHT      = WIDTH * 9 // 16
HALF_WIDTH  = WIDTH // 2
HALF_HEIGHT = HEIGHT // 2

MIN_SELECTION_WIDTH  = 16 # or 20, 10
MIN_SELECTION_HEIGHT = 9 # or 20, 10


if args['path']:
	stream = cv2.VideoCapture(args['path'])
	grabbed, frame = stream.read()
else:
	stream = WebcamVideoStream(src=args['camera']).start()
	frame = stream.read()

# if args['path']:
# 	stream = cv2.VideoCapture(args['path'])
# 	grabbed, frame = stream.read()
# else:
# 	stream = WebcamVideoStream(src=args['camera']).start()
# 	frame = stream.read()

frame = imutils.resize(frame, width=WIDTH)
prev_hash = hashlib.sha1(frame).hexdigest()
prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


pause_flag = False
tracking_processing_flag = False
force_init_flag = False
capture = None
tracking_window = {'x1': -1, 'y1': -1, 'x2': -1, 'y2': -1, 'dragging': False, 'start': False}
tracking_results = {'color_status': False, 'kcf_peak_value': 0.5}
motor_is_moving_flag = False
zoom_is_moving_flag = False
# current_zoom = 1
zooms = [1,2,4,8,12,16,20]
zoom_idx = 0
current_zoom = zooms[zoom_idx]

def motor_has_finished_moving(args):
	# print("Motor: End of Moving")
	global motor_is_moving_flag
	motor_is_moving_flag = False

def zoom_has_finished_moving(args):
	# zoom.stop_zooming()

	global zoom_is_moving_flag
	zoom_is_moving_flag = False
	print("Motor: End of Zooming")

def onmouse(event, x, y, flags, param):
	if event == cv2.EVENT_LBUTTONDOWN:
		param['dragging'] = True
		param['x1'] = x
		param['y1'] = y
		param['x2'] = x
		param['y2'] = y

	elif event == cv2.EVENT_MOUSEMOVE:
		if param['dragging'] == True:
			param['x2'] = x
			param['y2'] = y

	elif event == cv2.EVENT_LBUTTONUP:
		if param['dragging'] == True:
			xmin = min(param['x1'], param['x2'])
			ymin = min(param['y1'], param['y2'])
			xmax = max(param['x1'], param['x2'])
			ymax = max(param['y1'], param['y2'])

			param['x1'] = xmin
			param['y1'] = ymin
			param['x2'] = xmax
			param['y2'] = ymax
			# print(xmin, xmax, ymin, ymax)
			param['start'] = True
			param['dragging'] = False

if args["display"] is True:
	cv2.namedWindow('Tracking')
	cv2.setMouseCallback('Tracking', onmouse, tracking_window)

if args['serial']:
	motor = Motor(dev = args['serial'], baud = 115200)

if args['zoom']:
	zoom = Motor(dev = args['zoom'], baud = 115200)

if args['color'] is True:
	color_tracker = ColorTracker()
else:
	color_tracker = None

if args['kcf'] is True:
	kcf_tracker = KCFTracker(True, False, True) # hog, fixed_window, multiscale
else:
	kcf_tracker = None


fps = FPS().start()
while fps._numFrames < args["num_frames"]:
	if pause_flag is False:
		if args['path']:
			grabbed, frame = stream.read()
			if grabbed is not True:
				# print("End of Frame")
				break
		else:
			frame = stream.read()

		frame = imutils.resize(frame, width=WIDTH)
		frame_hash = hashlib.sha1(frame).hexdigest()

	if frame_hash != prev_hash:
		# print('o')

		if pause_flag is not True:
			if tracking_window['start'] == True:
				if((tracking_window['x2'] - tracking_window['x1']) > MIN_SELECTION_WIDTH) and ((tracking_window['y2'] - tracking_window['y1']) > MIN_SELECTION_HEIGHT):
					# print('selection is ok')

					if color_tracker:
						if color_tracker.init(frame, options=tracking_window):
							print('Red Found:', color_tracker.center)
						else:
							print('Red Not Found:', color_tracker.center)


					if kcf_tracker:
						# ix = tracking_window['x1']
						# iy = tracking_window['y1']
						# w = tracking_window['x2'] - tracking_window['x1']
						# h = tracking_window['y2'] - tracking_window['y1']
						# kcf_tracker.init([ix,iy,w,h], frame)

						# kcf_tracker.init(tracking_window, frame)

						kcf_tracker.x1 = tracking_window['x1']
						kcf_tracker.y1 = tracking_window['y1']
						kcf_tracker.x2 = tracking_window['x2']
						kcf_tracker.y2 = tracking_window['y2']

						kcf_tracker.init(frame)
						#if you use hog feature, there will be a short pause after you draw a first boundingbox, that is due to the use of Numba.

					tracking_processing_flag = True
				else:
					if args['serial'] and zoom_is_moving_flag is not True:
						centerX = (tracking_window['x1'] + tracking_window['x2']) // 2
						centerY = (tracking_window['y1'] + tracking_window['y2']) // 2
						center_to_x = centerX - HALF_WIDTH
						center_to_y = HALF_HEIGHT - centerY

						# print("Move to ({}, {})".format(center_to_x, center_to_y))
						motor_timer = Timer(1, motor_has_finished_moving, args = [False])
						(x_to, y_to, z_to, f_to) = motor.pixel_to_pulse(center_to_x, center_to_y, current_zoom, limit = False)
						motor.move(x = x_to, y = y_to, z = z_to, f = f_to, t = 1)
						motor_timer.start()
						motor_is_moving_flag = True

				capture = None
				tracking_window['start'] = False

			if tracking_processing_flag is True and motor_is_moving_flag is not True:
				tic = time.time()
				if color_tracker:
					if kcf_tracker:
						tracking_results['color_status'] = color_tracker.update(frame,  {'x1': kcf_tracker.x1, 'y1':kcf_tracker.y1, 'x2': kcf_tracker.x2, 'y2': kcf_tracker.y2})
					else:
						tracking_results['color_status'] = color_tracker.update(frame)
					if tracking_results['color_status'] is True:
						cv2.drawMarker(frame, tuple(color_tracker.center), (0,255,0))
					# else:
					# 	print("COLOR_TRACKER: Tracking fail")

				if kcf_tracker:
					if force_init_flag is not True:
						boundingbox, tracking_results['kcf_peak_value'], loc = kcf_tracker.update(frame)
						boundingbox = list(map(int, boundingbox))

						kcf_tracker.x1 = boundingbox[0]
						kcf_tracker.y1 = boundingbox[1]
						kcf_tracker.x2 = boundingbox[0] + boundingbox[2]
						kcf_tracker.y2 = boundingbox[1] + boundingbox[3]

						cv2.rectangle(frame,(kcf_tracker.x1,kcf_tracker.y1), (kcf_tracker.x2,kcf_tracker.y2), (0,255,255), 1)

						cX = (kcf_tracker.x1 + kcf_tracker.x2) // 2
						cY = (kcf_tracker.y1 + kcf_tracker.y2) // 2

						if tracking_results['color_status'] is True:
							diff = abs(color_tracker.center[0] - cX)
							if diff > boundingbox[2] // 6: # boundingbox[2] == width
								kcf_tracker.x1 = color_tracker.center[0] - boundingbox[2] // 2
								kcf_tracker.x2 = color_tracker.center[0] + boundingbox[2] // 2
								force_init_flag = True

							diff = abs(color_tracker.center[1] - cY)
							if diff < boundingbox[3] // 6: # boundingbox[3] == height
								kcf_tracker.y1 = int(color_tracker.center[1] - boundingbox[3] / 4)
								kcf_tracker.y2 = int(color_tracker.center[1] + boundingbox[3] * 3 / 4)
								# print(diff, kcf_tracker.x1, kcf_tracker.y1, kcf_tracker.x2, kcf_tracker.y2)
								force_init_flag = True
					else: # if force_init_flag is True:
						print('Force init...')
						force_init_flag = False
						kcf_tracker.init(frame)

				if args['serial'] and zoom_is_moving_flag is not True:
					motor_driving_flag = False

					if color_tracker and kcf_tracker is None:
						if tracking_results['color_status'] is True:
							cX, cY = color_tracker.center
							motor_driving_flag = True

					elif kcf_tracker:
						if tracking_results['kcf_peak_value'] > 0.2:
							cX = (kcf_tracker.x1 + kcf_tracker.x2) // 2
							cY = (kcf_tracker.y1 + kcf_tracker.y2) // 2
							motor_driving_flag = True
						else:
							print("KCF peak value is too low", tracking_results['kcf_peak_value'])
					else:
						cX = HALF_WIDTH
						cY = HALF_HEIGHT

					if motor_driving_flag is True:
						# cv2.drawMarker(frame, (cX, cY), (0,0,255))
						center_to_x = cX - HALF_WIDTH
						center_to_y = HALF_HEIGHT - cY
						# print("Distance from Center: ({}px, {}px)".format(center_to_x, center_to_y))

						# toc = time.time()
						# tic_to_toc = toc - tic

						# t_sec = tic_to_toc / 1000
						# t_usec = int(t_sec * 1000000)
						# current_time = datetime.datetime.now().time().isoformat()
						# print("Tracking duration: {:04.0f} ms @{}".format(1000*tic_to_toc, current_time))

						# 모터가 움직이는 동안 추적을 할 것인가 말것인가를 아래에서 결정
						# (1) 모터 움직일 동안 motor_is_moving_flag을 lock 시켜서 추적을 안하는 방안
						# (2) motor_is_moving_flag을 아예 셋팅하지 않고 추적을 계속하는 방안

						# case (1)
						# print("Move to ({}px, {}px)".format(center_to_x, center_to_y))
						t_sec = motor.track(center_to_x, center_to_y, current_zoom)

						if t_sec > 0:
							motor_is_moving_flag = True
							motor_timer = Timer(t_sec, motor_has_finished_moving, args = [False])
							motor_timer.start()

						# or case (2)
						# (t_sec, t_usec) = motor.track(center_to_x, center_to_y, current_zoom)

					# d = max(abs(x_to), abs(y_to))
					# dy = HEIGHT - cY
					# dx = abs(WIDTH/2 - cX)
					# # d = int(math.sqrt(dx**2 + dy**2))
					# d = dy
					# print(d, ":", kcf_tracker._scale)

					# if tracking_results['kcf_peak_value'] < 0.3:
						# print(kcf_tracker.location, kcf_tracker.locations[-1], loc)
						# adj_x = int(boundingbox[2] * loc[0]/4)
						# adj_y = int(boundingbox[3] * loc[1]/4)
						# if kcf_tracker.direction['y'] == 'down':
						# 	kcf_tracker.x1 -= adj_y
						# 	kcf_tracker.y1 -= adj_y
						# 	kcf_tracker.x2 += adj_y
						# 	kcf_tracker.y2 += adj_y
						# else:
						# 	kcf_tracker.x1 += adj_y
						# 	kcf_tracker.y1 += adj_y
						# 	kcf_tracker.x2 -= adj_y
						# 	kcf_tracker.y2 -= adj_y


						# cv2.rectangle(frame, (kcf_tracker.x1, kcf_tracker.y1), (kcf_tracker.x2, kcf_tracker.y2), (0,0,255), 1)
						# kcf_tracker.init(frame)
						# pause_flag = True







			cv2.line(frame, (HALF_WIDTH, 0), (HALF_WIDTH, WIDTH), (200, 200, 200), 0)
			cv2.line(frame, (0, HALF_HEIGHT), (WIDTH, HALF_HEIGHT), (200, 200, 200), 0)


		if args["display"] is True:
			if tracking_window['dragging'] == True:
				pt1 = (tracking_window['x1'], tracking_window['y1'])
				pt2 = (tracking_window['x2'], tracking_window['y2'])

				if capture is None:
					capture = np.copy(frame)

				frame = np.copy(capture)
				cv2.rectangle(frame, pt1, pt2, (0, 255, 0,), 1)
				cv2.imshow("Tracking", frame)

			# pause가 아닌 상태에서 Tracking window 보이기(당연),
			# 그런데 pause 일때 굳이 동작 않도록 처리한 이유는? => pause 일때 마우스 조작이 일어나는 경우에 대처하기 위해, 즉, 다른곳에서 윈도우 처리
			if pause_flag is False: cv2.imshow("Tracking", frame)

		key = cv2.waitKey(1)
		if key == 27 or key == ord('q'):
			break
		elif key == ord(' '):
			pause_flag = not pause_flag
		elif key == ord('s'):
			tracking_processing_flag = False
		elif key == ord('i'):
			force_init_flag = True
		elif key == 65362: # 'up', 63232 for Mac
			if zoom_is_moving_flag is not True and current_zoom < 20:
				zoom_idx += 1
				current_zoom = zooms[zoom_idx]
				zoom.zoom_to(current_zoom)
				zoom_is_moving_flag = True
				zoom_timer = Timer(1, zoom_has_finished_moving, args = [False])
				zoom_timer.start()
		elif key == 65364: # 'down', 63233 for Mac
			if zoom_is_moving_flag is not True and current_zoom > 1:
				zoom_idx -= 1
				current_zoom = zooms[zoom_idx]
				zoom.zoom_to(current_zoom)
				zoom_is_moving_flag = True
				zoom_timer = Timer(1, zoom_has_finished_moving, args = [False])
				zoom_timer.start()

		# elif key == 65362: # 'up', 63232 for Mac
		# 	if zoom_is_moving_flag is not True: # and current_zoom < 17:
		# 		current_zoom += 1
		# 		zoom_timer = Timer(0.171875, zoom_has_finished_moving, args = [False])
		# 		zoom.zoom(current_zoom, direction='in')
		# 		zoom_timer.start()
		# 		zoom_is_moving_flag = True
		# 		print('Zoom to:', current_zoom)
		# elif key == 65364: # 'down', 63233 for Mac
		# 	if zoom_is_moving_flag is not True: # and current_zoom > 1:
		# 		current_zoom -= 1
		# 		zoom_timer = Timer(0.171875, zoom_has_finished_moving, args = [False])
		# 		zoom.zoom(current_zoom, direction='out')
		# 		zoom_timer.start()
		# 		zoom_is_moving_flag = True
		# 		print('Zoom to:', current_zoom)
		elif key == 65361: # 'left', 63234 for Mac
			if zoom_is_moving_flag is not True:
				zoom_idx = 0
				current_zoom = zooms[zoom_idx]
				zoom.zoom_x1()
				zoom_is_moving_flag = True
				zoom_timer = Timer(2.5, zoom_has_finished_moving, args = [False])
				zoom_timer.start()
		elif key == 65363: # 'right', 63235 for Mac
			if zoom_is_moving_flag is not True:
				zoom_idx = 6
				current_zoom = zooms[zoom_idx]
				zoom.zoom_x20()
				zoom_is_moving_flag = True
				zoom_timer = Timer(2.5, zoom_has_finished_moving, args = [False])
				zoom_timer.start()

		elif key == ord('d'):
			print("Degree:", motor.sum_of_x_degree, motor.sum_of_y_degree)

		# pause가 아니면 prev_hash를 갱신함 (당연),
		# 반대로 pause일때는 갱신하지 않음으로 prev_hash와 frame_hash를 불일치 시킴. Why? pause에도 key 입력 루틴을 실행하기 위한 의도적인 조치임
		if pause_flag is False: prev_hash = frame_hash
		fps.update()

	# else: # 같은 frame을 반복해서 읽는 경우
	# 	print('x')

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
if args['path']:
	stream.release()
else:
	stream.stop()
