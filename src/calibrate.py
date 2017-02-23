# import the necessary packages

from __future__ import print_function
import sys
PY3 = sys.version_info[0] == 3
if PY3:
    xrange = range


import cv2
import numpy as np
import imutils

from motor.motor import Motor

import math
from threading import Timer
import time
import datetime

from utils import common
from utils import util

import os
import glob
import re

import argparse

ap = argparse.ArgumentParser()

ap.add_argument("-c", "--camera", help = "camera number")
ap.add_argument("-p", "--path", help = "path to video file")
ap.add_argument("-d", "--display", action="store_true", help="show display")
ap.add_argument("-m", "--motor", help = "path to motor device")
ap.add_argument("-z", "--zoom", help = "path to zoom control port")
ap.add_argument("-w", "--width", type=int, default=640, help="screen width in px")

args = vars(ap.parse_args())
# print("[INFO] Command: ", args)

FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080

WIDTH       = args['width']  # 640x360, 1024x576, 1280x720, 1920x1080
HEIGHT      = WIDTH * 9 // 16
HALF_WIDTH  = WIDTH // 2
HALF_HEIGHT = HEIGHT // 2

MIN_SELECTION_WIDTH  = 16 # or 20, 10
MIN_SELECTION_HEIGHT = 9 # or 20, 10

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
            # print("[MOUSE]", xmin, xmax, ymin, ymax)
            param['start'] = True
            param['dragging'] = False

if args['path']:
    stream = cv2.VideoCapture(args['path'])
    FRAME_HEIGHT = int(stream.get(4))
    FRAME_WIDTH = int(stream.get(3))
else:
    stream = cv2.VideoCapture(args['camera'])
    stream.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    stream.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

if args['motor']:
    motor = Motor(dev = args['motor'], baud = 115200, screen_width = WIDTH)
    motor_flag = True
else:
    motor = None
    motor_flag = False

if args['zoom']:
    zoom = Motor(dev = args['zoom'], baud = 115200, screen_width = WIDTH)
    zoom.zoom_x1()
    zoom_flag = True
else:
    zoom = None
    zoom_flag = False

cv2.namedWindow('Calibrate')
cv2.moveWindow('Calibrate', 0, 0)
tracking_window = {'x1': -1, 'y1': -1, 'x2': -1, 'y2': -1, 'dragging': False, 'start': False}
cv2.setMouseCallback('Calibrate', onmouse, tracking_window)

capture = None

pause_flag = False
color_select_flag = False
set_preset_flag = False
set_move_flag = True


# Read the first frame
grabbed, frame = stream.read()
frame = imutils.resize(frame, width=WIDTH)

def nothing(x):
    pass

# Initialize to check if HSV min/max value changes
hMin = sMin = vMin = hMax = sMax = vMax = 0
phMin = psMin = pvMin = phMax = psMax = pvMax = 0

while True:
    if pause_flag is False:
        grabbed, frame = stream.read()
        frame = imutils.resize(frame, width = WIDTH)
        # print("[INFO] frame.shape: ", frame.shape)

        if color_select_flag is True:
            # get current positions of all trackbars
            hMin = cv2.getTrackbarPos('HMin','Tracking')
            sMin = cv2.getTrackbarPos('SMin','Tracking')
            vMin = cv2.getTrackbarPos('VMin','Tracking')

            hMax = cv2.getTrackbarPos('HMax','Tracking')
            sMax = cv2.getTrackbarPos('SMax','Tracking')
            vMax = cv2.getTrackbarPos('VMax','Tracking')

            # Set minimum and max HSV values to display
            lower = np.array([hMin, sMin, vMin])
            upper = np.array([hMax, sMax, vMax])

            # Create HSV Image and threshold into a range.
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            # hsv[:,:,2] = cv2.equalizeHist(hsv[:,:,2])
            mask = cv2.inRange(hsv, lower, upper)
            frame = cv2.bitwise_and(frame, frame, mask= mask)
            # cv2.imshow('Output', output)

            # Print if there is a change in HSV value
            if( (phMin != hMin) | (psMin != sMin) | (pvMin != vMin) | (phMax != hMax) | (psMax != sMax) | (pvMax != vMax) ):
                print("(hMin = {} , sMin = {}, vMin = {}), (hMax = {} , sMax = {}, vMax = {})".format(hMin , sMin , vMin, hMax, sMax , vMax))
                phMin = hMin
                psMin = sMin
                pvMin = vMin
                phMax = hMax
                psMax = sMax
                pvMax = vMax


        if tracking_window['start'] == True:
            if((tracking_window['x2'] - tracking_window['x1']) > MIN_SELECTION_WIDTH) and ((tracking_window['y2'] - tracking_window['y1']) > MIN_SELECTION_HEIGHT):
                width = tracking_window['x2'] - tracking_window['x1']
                height = tracking_window['y2'] - tracking_window['y1']
                area = width * height
                print("[Debug] Area:{}, Width: {}, Height: {} @ x{}".format(area, width, height, current_zoom))
                # roi = frame[tracking_window['y1']:tracking_window['y2'],tracking_window['x1']:tracking_window['x2']]
                # blur = cv2.GaussianBlur(roi, (123, 123), 0)
                # cv2.imshow('ROI and Blur', np.hstack([roi, blur]))
                # blur = cv2.GaussianBlur(frame, (123, 123), 0)
                # cv2.imshow('Frame and Blur', np.hstack([frame, blur]))

            else:
                if False: #color_select_flag is True:
                    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    roi = frame[tracking_window['x1']-10:tracking_window['x1']+10, tracking_window['y1']-10:tracking_window['y1']+10]
                    (means, stds) = cv2.meanStdDev(roi)
                    # print("[INFO] HSV means: {}, stds: {}".format(means, stds))
                    print("H:{:03d}±{:02d}, S:{:03d}±{:02d}, V:{:03d}±{:02d}".
                    format(int(means[0,0]), int(stds[0,0]), int(means[1,0]), int(stds[1,0]), int(means[2,0]), int(stds[2,0])))

                    #
                    # lower = cv2.subtract(np.uint8([means]), np.uint8([stds]))
                    # upper = cv2.add(np.uint8([means]), np.uint8([stds]))
                    #
                    # # upper[0][0] =  upper[0][0] % 180
                    # print("Lower: {}, Upper: {}".format(lower, upper))
                elif motor_flag and motor.is_moving is False:
                    centerX = (tracking_window['x1'] + tracking_window['x2']) // 2
                    centerY = (tracking_window['y1'] + tracking_window['y2']) // 2
                    center_to_x = centerX - HALF_WIDTH
                    center_to_y = HALF_HEIGHT - centerY # new controller 2월 1일

                    if zoom_flag is False:
                        motor.move_to(center_to_x, center_to_y)
                    elif zoom_flag and zoom.is_zooming is not True:
                        motor.move_to(center_to_x, center_to_y, zoom.current_zoom)

            capture = None
            tracking_window['start'] = False

    if args["display"] is True:
        cv2.line(frame, (HALF_WIDTH, 0), (HALF_WIDTH, WIDTH), (200, 200, 200), 0)
        cv2.line(frame, (0, HALF_HEIGHT), (WIDTH, HALF_HEIGHT), (200, 200, 200), 0)

        if set_preset_flag:
            util.draw_str(frame, (20, 20), 'Preset setting mode')



        if True: #set_preset_flag is True:
            x = 4
            k = x * 1
            x1 = int(HALF_WIDTH-(WIDTH/k))
            y1 = int(HALF_HEIGHT-(HEIGHT/k))
            x2 = int(HALF_WIDTH+(WIDTH/k))
            y2 = int(HALF_HEIGHT+(HEIGHT/k))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 0)

            # k = x * 2
            # x1 = int(HALF_WIDTH-(WIDTH/k))
            # y1 = int(HALF_HEIGHT-(HEIGHT/k))
            # x2 = int(HALF_WIDTH+(WIDTH/k))
            # y2 = int(HALF_HEIGHT+(HEIGHT/k))
            # cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 0)
            #
            # k = x * 4
            # x1 = int(HALF_WIDTH-(WIDTH/k))
            # y1 = int(HALF_HEIGHT-(HEIGHT/k))
            # x2 = int(HALF_WIDTH+(WIDTH/k))
            # y2 = int(HALF_HEIGHT+(HEIGHT/k))
            # cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 0)
            #
            # k = x * 8
            # x1 = int(HALF_WIDTH-(WIDTH/k))
            # y1 = int(HALF_HEIGHT-(HEIGHT/k))
            # x2 = int(HALF_WIDTH+(WIDTH/k))
            # y2 = int(HALF_HEIGHT+(HEIGHT/k))
            # cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 0)
            #
            # k = x * 16
            # x1 = int(HALF_WIDTH-(WIDTH/k))
            # y1 = int(HALF_HEIGHT-(HEIGHT/k))
            # x2 = int(HALF_WIDTH+(WIDTH/k))
            # y2 = int(HALF_HEIGHT+(HEIGHT/k))
            # cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 0)

        if tracking_window['dragging'] == True:
            pt1 = (tracking_window['x1'], tracking_window['y1'])
            pt2 = (tracking_window['x2'], tracking_window['y2'])

            if capture is None:
                capture = np.copy(frame)

            frame = np.copy(capture)
            cv2.rectangle(frame, pt1, pt2, (0, 255, 0,), 1)
            cv2.imshow("Calibrate", frame)

        # pause가 아닌 상태에서 Tracking window 보이기(당연),
        # 그런데 pause 일때 굳이 동작 않도록 처리한 이유는? => pause 일때 마우스 조작이 일어나는 경우에 대처하기 위해, 즉, 다른곳에서 윈도우 처리
        if pause_flag is False:
            cv2.imshow("Calibrate", frame)

    key = cv2.waitKey(1)
    if key == 27 or key == ord('q'):
        break
    elif key == ord(' '):
        pause_flag = not pause_flag
    elif key == ord('p'):
        set_preset_flag = not set_preset_flag
    elif key == ord('m'):
        set_move_flag = not set_move_flag
    elif key >= ord('1') and key <= ord('5'):
        preset = key - 48
        if set_preset_flag is True:
            zoom.set_preset(preset)
            current_zoom = {v:k for k, v in zoom.zoom_to_preset.items()}[preset]
            zoom.current_zoom = current_zoom
        else:
            zoom.get_preset(preset)
            current_zoom = {v:k for k, v in zoom.zoom_to_preset.items()}[preset]
            zoom.current_zoom = current_zoom
    elif key%256 == 82: # 'up': 65362 for Ubuntu, 63232 for Mac
        if set_preset_flag:
            if zoom_flag and zoom.is_zooming is not True:
                next_zoom = zoom.find_next_zoom(dir='in')
                if next_zoom != zoom.current_zoom:
                    zoom.zoom_to(next_zoom, dur=0.1)
        else:
            if motor_flag and motor.is_moving is False:
                motor.move_to(0, 1, zoom.current_zoom)
    elif key%256 == 84: # 'down': 65364 for Ubuntu, 63233 for Mac
        if set_preset_flag:
            if zoom_flag and zoom.is_zooming is not True:
                next_zoom = zoom.find_next_zoom(dir='out')
                if next_zoom != zoom.current_zoom:
                    zoom.zoom_to(next_zoom, dur=0.1)
        else:
            if motor_flag and motor.is_moving is False:
                motor.move_to(0, -1, zoom.current_zoom)
    elif key%256 == 81: # 'left': 65361 for Ubuntu, 63234 for Mac
        if set_preset_flag:
            if zoom_flag and zoom.is_zooming is not True:
                zoom.zoom('out')
        else:
            if motor_flag and motor.is_moving is False:
                motor.move_to(-1, 0, zoom.current_zoom)

    elif key%256 == 83: # 'right': 65363 for Ubuntu, 63235 for Mac
        if set_preset_flag:
            if zoom_flag and zoom.is_zooming is not True:
                zoom.zoom('in')
        else:
            if motor_flag and motor.is_moving is False:
                motor.move_to(1, 0, zoom.current_zoom)
    elif key == ord('c'):
        color_select_flag = not color_select_flag
        if color_select_flag is True:
            # create trackbars for color change
            cv2.createTrackbar('HMin','Tracking',0,179,nothing) # Hue is from 0-179 for Opencv
            cv2.createTrackbar('SMin','Tracking',0,255,nothing)
            cv2.createTrackbar('VMin','Tracking',0,255,nothing)
            cv2.createTrackbar('HMax','Tracking',0,179,nothing)
            cv2.createTrackbar('SMax','Tracking',0,255,nothing)
            cv2.createTrackbar('VMax','Tracking',0,255,nothing)

            # Set default value for MAX HSV trackbars.
            cv2.setTrackbarPos('HMax', 'Tracking', 179)
            cv2.setTrackbarPos('SMax', 'Tracking', 255)
            cv2.setTrackbarPos('VMax', 'Tracking', 255)
        else:
            cv2.destroyWindow('Tracking')
            cv2.namedWindow('Tracking')
            cv2.moveWindow('Tracking', 0, 0)

# do a bit of cleanup
cv2.destroyAllWindows()
stream.release()
