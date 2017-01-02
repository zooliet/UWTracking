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

from motor import Motor

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
# ap.add_argument("--camera", type=int, default=0, help = "camera number")
ap.add_argument("-c", "--camera", type=int, default=0, help = "camera number")
ap.add_argument("-p", "--path", help = "path to video file")
ap.add_argument("-d", "--display", action="store_true", help="Show display")
ap.add_argument("-s", "--serial", help = "path to serial device")
ap.add_argument("-z", "--zoom", help = "path to zoom control port")

args = vars(ap.parse_args())
print("[INFO] Command: ", args)


WIDTH       = 640  # 640x360, 1024x576, 1280x720, 1920x1080
HEIGHT      = WIDTH * 9 // 16
HALF_WIDTH  = WIDTH // 2
HALF_HEIGHT = HEIGHT // 2

MIN_SELECTION_WIDTH  = 16 # or 20, 10
MIN_SELECTION_HEIGHT = 9 # or 20, 10

if args['path']:
    stream = cv2.VideoCapture(args['path'])
else:
    stream = cv2.VideoCapture(args['camera'])
    grabbed, frame = stream.read()

stream.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
stream.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

pause_flag = False
capture = None
tracking_window = {'x1': -1, 'y1': -1, 'x2': -1, 'y2': -1, 'dragging': False, 'start': False}
motor_is_moving_flag = False
zoom_is_moving_flag = False
fifo_enable_flag = False

zooms = [1,2,4,8,16]
zoom_idx = 0
current_zoom = zooms[zoom_idx]

def motor_has_finished_moving(args):
    global motor_is_moving_flag
    motor_is_moving_flag = False
    # print("[MOTOR] End of Moving")

def zoom_has_finished_moving(args):
    # zoom.stop_zooming()

    global zoom_is_moving_flag
    zoom_is_moving_flag = False
    # print("[ZOOM] End of Zooming")

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

if args["display"] is True:
    cv2.namedWindow('Tracking')
    cv2.setMouseCallback('Tracking', onmouse, tracking_window)

if args['serial']:
    motor = Motor(dev = args['serial'], baud = 115200)
else:
    motor = None

if args['zoom']:
    zoom = Motor(dev = args['zoom'], baud = 115200)
    zoom.zoom_x1()
else:
    zoom = None


while True:
    if pause_flag is False:
        grabbed, frame = stream.read()
        # frame = imutils.resize(frame, width = WIDTH)
        # print("[INFO] frame.shape: ", frame.shape)

        if tracking_window['start'] == True:
            if((tracking_window['x2'] - tracking_window['x1']) > MIN_SELECTION_WIDTH) and ((tracking_window['y2'] - tracking_window['y1']) > MIN_SELECTION_HEIGHT):
                width = tracking_window['x2'] - tracking_window['x1']
                height = tracking_window['y2'] - tracking_window['y1']
                area = width * height
                print("[Debug] Area:{}, Width: {}, Height: {} @ x{}".format(area, width, height, current_zoom))

            elif args['serial'] and zoom_is_moving_flag is not True:
                centerX = (tracking_window['x1'] + tracking_window['x2']) // 2
                centerY = (tracking_window['y1'] + tracking_window['y2']) // 2
                center_to_x = HALF_WIDTH - centerX
                center_to_y = centerY - HALF_HEIGHT

                motor_timer = Timer(1, motor_has_finished_moving, args = [False])
                (x_to, y_to, z_to, f_to) = motor.pixel_to_pulse(center_to_x, center_to_y, current_zoom, limit = False)
                motor.move(x = x_to, y = y_to, z = z_to, f = f_to, t = 1)
                motor_timer.start()
                motor_is_moving_flag = True

            capture = None
            tracking_window['start'] = False

    if args["display"] is True:
        cv2.line(frame, (HALF_WIDTH, 0), (HALF_WIDTH, WIDTH), (200, 200, 200), 0)
        cv2.line(frame, (0, HALF_HEIGHT), (WIDTH, HALF_HEIGHT), (200, 200, 200), 0)

        if True:
            x = 8
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
            cv2.imshow("Tracking", frame)

        # pause가 아닌 상태에서 Tracking window 보이기(당연),
        # 그런데 pause 일때 굳이 동작 않도록 처리한 이유는? => pause 일때 마우스 조작이 일어나는 경우에 대처하기 위해, 즉, 다른곳에서 윈도우 처리
        if pause_flag is False:
            cv2.imshow("Tracking", frame)
            if fifo_enable_flag is True:
                fifo.write(frame)

    key = cv2.waitKey(1)
    if key == 27 or key == ord('q'):
        break
    elif key == ord(' '):
        pause_flag = not pause_flag
    elif key == ord('f'):
        fifo_enable_flag = not fifo_enable_flag
        if fifo_enable_flag is True:
            print('[FIFO] Enabled')
            fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
            if not os.path.exists('captures'):
                os.makedirs('captures')
            files = glob.glob('captures/capture-*.mkv')
            if len(files) > 0:
                files.sort()
                last_file = files[-1]
                last_num = re.findall(r"[0-9]{4}", last_file)[0]
                last_num = int(last_num)
                pic_num = last_num + 1
            else:
                pic_num = 0

            file_name =  "captures/capture-{:04d}.mkv".format(pic_num)
            fifo = cv2.VideoWriter(file_name, fourcc, 20.0, (frame.shape[1], frame.shape[0]))
        else:
            print('[FIFO] Disabled')

    elif key >= ord('1') and key <= ord('5'):
        preset = key - 48
        zoom.set_preset(preset)
    elif key == 65362: # 'up', 63232 for Mac
        if zoom_is_moving_flag is not True and current_zoom < 16:
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
    elif key == 65361: # 'left', 63234 for Mac
        if zoom_is_moving_flag is not True:
            # print("[ZOOM] to 1")
            # zoom_idx = 0
            # current_zoom = zooms[zoom_idx]
            # zoom.zoom_x1()
            # zoom_is_moving_flag = True
            # zoom_timer = Timer(2.5, zoom_has_finished_moving, args = [False])
            # zoom_timer.start()
            zoom.zoom('out')
    elif key == 65363: # 'right', 63235 for Mac
        if zoom_is_moving_flag is not True:
            # print("[ZOOM] to 20")
            # zoom_idx = 6
            # current_zoom = zooms[zoom_idx]
            # zoom.zoom_x20()
            # zoom_is_moving_flag = True
            # zoom_timer = Timer(2.5, zoom_has_finished_moving, args = [False])
            # zoom_timer.start()
            zoom.zoom('in')
