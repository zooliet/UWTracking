# -*- coding: utf-8 -*-

from __future__ import print_function
import sys
PY3 = sys.version_info[0] == 3
if PY3:
    xrange = range

import cv2
import numpy as np
import imutils

from trackers.kcf.kcf_tracker import KCFTracker
from trackers.color.color_tracker import ColorTracker
from trackers.motion.motion_tracker import MotionTracker
from trackers.dlib.dlib_tracker import DLIBTracker
from trackers.cmt.cmt_tracker import CMTTracker

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

import redis
import json
from controller import redis_agent

# construct the argument parse and parse the arguments
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--camera", help = "camera number")
ap.add_argument("-p", "--path", help = "path to video file")

ap.add_argument("-n", "--num-frames", type=int, default=10000000, help="# of frames to loop over")
ap.add_argument("-d", "--display", action="store_true", help="show display")

ap.add_argument("-m", "--motor", help = "path to motor device")
ap.add_argument("-z", "--zoom", help = "path to zoom control port")
ap.add_argument("-w", "--width", type=int, default=639, help="screen width in px")

ap.add_argument("--kcf", action="store_true", help="Enable KCF tracking")
ap.add_argument("--dlib", action="store_true", help="Enable dlib tracking")
ap.add_argument("--cmt", action="store_true", help="Enable CMT tracking")

ap.add_argument("--color", action="store_true", help="Enable color subtracking")
ap.add_argument("--motion", action="store_true", help="Enable Motion subtracking")
ap.add_argument("--autozoom", action="store_true", help="Enable automatic zoom control")

ap.add_argument("--view", help = "select the view") # front, rear, side
ap.add_argument("--gui", action="store_true", help="Enable GUI control")

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

def tracking_processing_delay(args):
    global tracking_is_waiting
    tracking_is_waiting = False
    # current_time = datetime.datetime.now().time().isoformat()
    # print("[Tracking] End of Waiting: {}".format(current_time)) # hl1sqi

if args["display"] is True:
    if args['view'] == 'front':
        title = '전면'
        win_x = 1280
        win_y = 0
        channel_name = 'uwtec:front'
        capture_name = 'front'
    elif args['view'] == 'rear':
        title = '후면'
        win_x = 0
        win_y = 0
        channel_name = 'uwtec:rear'
        capture_name = 'rear'
    elif args['view'] == 'side':
        title = '측면'
        win_x = 640
        win_y = 0
        channel_name = 'uwtec:side'
        capture_name = 'side'
    else:
        title = '카메라'
        win_x = 0
        win_y = 0
        channel_name = 'uwtec:camera'
        capture_name = 'camera'

if args['path']:
    stream = cv2.VideoCapture(args['path'])
    FRAME_HEIGHT = int(stream.get(4))
    FRAME_WIDTH = int(stream.get(3))
else:
    if args['camera'].isdigit():
        stream = cv2.VideoCapture(int(args['camera']))
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
    autozoom_flag = True if args['autozoom'] else False
else:
    zoom = None
    zoom_flag = False
    autozoom_flag = False

if args['gui']:
    gui_flag = True
    r = redis.Redis()
    redis_agent = redis_agent.RedisAgent(r, [channel_name])
    redis_agent.start()
else:
    gui_flag = False


cv2.namedWindow(title)
cv2.moveWindow(title, win_x, win_y)
tracking_window = {'x1': -1, 'y1': -1, 'x2': -1, 'y2': -1, 'dragging': False, 'start': False}
cv2.setMouseCallback(title, onmouse, tracking_window)

capture = None

pause_flag = False
tracking_processing_flag = False
show_lap_time_flag = False
upscale_flag = False
wide_zoom_flag = False
fifo_enable_flag = False
limit_setup_flag = False
tracking_is_waiting = True

tic = time.time()
toc = time.time()

# Read the first frame
grabbed, frame_full = stream.read()
frame = imutils.resize(frame_full, width=640)
prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

grabbed, frame_full = stream.read()
frame = imutils.resize(frame_full, width=640)
prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# cv2.imshow('T', frame)
# cv2.waitKey(3000)


if args['kcf'] is True:
    kcf_tracker = KCFTracker(True, False, True) # hog, fixed_window, multiscale
else:
    kcf_tracker = None

if args['dlib'] is True:
    dlib_tracker = DLIBTracker()
else:
    dlib_tracker = None

# if args['cmt'] is True:
#     cmt_tracker = CMTTracker(True, False, cmt_detector_threshold = 50, best_effort = False) # estimate_scale, estimate_rotation
# elif args['cmt_alone'] is True:
#     cmt_tracker = CMTTracker(True, False, cmt_detector_threshold = 50, best_effort = True) # estimate_scale, estimate_rotation

if args['cmt'] is True:
    cmt_tracker = CMTTracker(True, False, cmt_detector_threshold = 50, best_effort = False) # estimate_scale, estimate_rotation
    gray0 = cv2.GaussianBlur(prev_gray, (3, 3), 0)

    for x in range(100, 0, -10):
        detector = cv2.BRISK_create(x, 3, 3.0)
        keypoints = detector.detect(gray0)
        # print("[CMT] {} => {}".format(x, len(keypoints)))
        cmt_detector_threshold = x
        if len(keypoints) > cmt_tracker.MIN_NUM_OF_KEYPOINTS_FOR_BRISK_THRESHOLD:
            break
    print("[CMT] BRISK threshold is set to {} with {} keypoints".format(x, len(keypoints)))
    cmt_tracker.detector = detector
    cmt_tracker.descriptor = detector
else:
    cmt_tracker = None

if args['color'] is True:
    color_tracker = ColorTracker(width = WIDTH, height = HEIGHT)
else:
    color_tracker = None

if args['motion'] is True:
    motion_tracker = MotionTracker()
else:
    motion_tracker = None

while True:
    if pause_flag is False:
        grabbed, frame_full = stream.read()
        if grabbed is not True:
            # print("End of Frame")
            break

        frame = imutils.resize(frame_full, width=WIDTH)
        frame_draw = np.copy(frame)

        if tracking_window['start'] == True:
            if((tracking_window['x2'] - tracking_window['x1']) > MIN_SELECTION_WIDTH) and ((tracking_window['y2'] - tracking_window['y1']) > MIN_SELECTION_HEIGHT):
                selected_width = tracking_window['x2'] - tracking_window['x1']
                selected_height = tracking_window['y2'] - tracking_window['y1']

                if zoom_flag:
                    selected_width = int(selected_width / zoom.current_zoom)
                    selected_height = int(selected_height / zoom.current_zoom)
                # print("[KCF] User selected width {} and height {}".format(selected_width, selected_height) )

                if kcf_tracker:
                    kcf_tracker.x1 = tracking_window['x1']
                    kcf_tracker.y1 = tracking_window['y1']
                    kcf_tracker.x2 = tracking_window['x2']
                    kcf_tracker.y2 = tracking_window['y2']

                    #if you use hog feature, there will be a short pause after you draw a first boundingbox, that is due to the use of Numba.
                    kcf_tracker.init(frame)

                    if motor_flag and limit_setup_flag is False:
                        tracking_processing_flag = False
                    else:
                        tracking_processing_flag = True
                        tracking_is_waiting = True
                        tracking_timer = Timer(3, tracking_processing_delay, args = [False])
                        tracking_timer.start()

                    # if color_tracker:
                    #     color_tracker.init(0)
                    # elif motion_tracker:
                    #     motion_tracker.init(0)
                    #     if motor_flag:
                    #         motor.stop_moving = False
                elif dlib_tracker:
                    dlib_tracker.x1 = tracking_window['x1']
                    dlib_tracker.y1 = tracking_window['y1']
                    dlib_tracker.x2 = tracking_window['x2']
                    dlib_tracker.y2 = tracking_window['y2']

                    dlib_tracker.init(frame)
                    if motor_flag and limit_setup_flag is False:
                        tracking_processing_flag = False
                    else:
                        tracking_processing_flag = True
                        tracking_is_waiting = True
                        tracking_timer = Timer(3, tracking_processing_delay, args = [False])
                        tracking_timer.start()

                elif cmt_tracker:
                    cmt_tracker.x1 = tracking_window['x1']
                    cmt_tracker.y1 = tracking_window['y1']
                    cmt_tracker.x2 = tracking_window['x2']
                    cmt_tracker.y2 = tracking_window['y2']

                    cmt_tracker.init(frame)
                    if cmt_tracker.num_initial_keypoints == 0:
                        print('[CMT] No keypoints found in selection')
                        if tracking_processing_flag == True: # reinitialize case
                            tracking_window['start'] = True # 강제로 초기화를 다시하는 효과
                        else:
                            tracking_processing_flag = True
                            tracking_is_waiting = True
                            tracking_timer = Timer(3, tracking_processing_delay, args = [False])
                            tracking_timer.start()
                    else:
                        # print("[CMT] num_selected_keypoints is {}".format(cmt_tracker.num_initial_keypoints))
                        if motor_flag and limit_setup_flag is False:
                            tracking_processing_flag = False
                        else:
                            tracking_processing_flag = True
                            tracking_is_waiting = True
                            tracking_timer = Timer(3, tracking_processing_delay, args = [False])
                            tracking_timer.start()

            elif motor_flag and ((tracking_processing_flag and motor.is_moving is False) or tracking_processing_flag is False ): #  and motor.is_moving is not True:  hl1sqi
                centerX = (tracking_window['x1'] + tracking_window['x2']) // 2
                centerY = (tracking_window['y1'] + tracking_window['y2']) // 2
                center_to_x = centerX - HALF_WIDTH
                center_to_y = HALF_HEIGHT - centerY

                if zoom_flag is False:
                    motor.move_to(center_to_x, center_to_y)
                elif zoom_flag and zoom.is_zooming is not True:
                    motor.move_to(center_to_x, center_to_y, zoom.current_zoom)

            capture = None
            tracking_window['start'] = False

        elif tracking_processing_flag is True:
            if show_lap_time_flag is True: # 'l' key
                current_time = datetime.datetime.now().time().isoformat()
                toc = time.time()
                print("[INFO] Tracking duration: {:04.0f} ms @{}".format(1000*(toc-tic), current_time))
                tic = toc

            if kcf_tracker:
                if kcf_tracker.force_init_flag is True:
                    print('[KCF] Force init')
                    kcf_tracker.init(frame)
                    kcf_tracker.force_init_flag = False
                elif kcf_tracker.enable:
                    boundingbox, loc = kcf_tracker.update(frame)
                    boundingbox = list(map(int, boundingbox))

                    # 이탈 정도(0.25), motion_tracker.interval, waitKey(x) 조정 필요
                    if kcf_tracker.peak_value < 0.25:
                        print('[KCF] Disabled: peak value({:.02f}) is too low'.format(kcf_tracker.peak_value))
                        kcf_tracker.enable = False
                        wide_zoom_flag = True

                    else:
                        wide_zoom_flag = False # just in case

                        kcf_tracker.x1 = boundingbox[0]
                        kcf_tracker.y1 = boundingbox[1]
                        kcf_tracker.x2 = boundingbox[0] + boundingbox[2]
                        kcf_tracker.y2 = boundingbox[1] + boundingbox[3]
                        kcf_tracker.center = ((kcf_tracker.x1 + kcf_tracker.x2) // 2, (kcf_tracker.y1 + kcf_tracker.y2) // 2)

                        kcf_tracker.prev_widths = np.append(kcf_tracker.prev_widths, boundingbox[2])
                        kcf_tracker.prev_heights = np.append(kcf_tracker.prev_heights, boundingbox[3])

                        if kcf_tracker.prev_widths.shape[0] > kcf_tracker.PREV_HISTORY_SIZE: # 10
                            kcf_tracker.prev_widths = np.delete(kcf_tracker.prev_widths, (0), axis=0)
                            kcf_tracker.prev_heights = np.delete(kcf_tracker.prev_heights, (0), axis=0)

                        kcf_tracker.mean_width = np.round(np.mean(kcf_tracker.prev_widths)).astype(np.int)
                        kcf_tracker.mean_height = np.round(np.mean(kcf_tracker.prev_heights)).astype(np.int)

                        # str = "{}x{}({}x{})".format(kcf_tracker.mean_width, kcf_tracker.mean_height, selected_width, selected_height)
                        # util.draw_str(frame_draw, (20, 20), str)
                        # util.draw_str(frame_draw, (550, 20), 'Tracking')
                        cv2.rectangle(frame_draw,(kcf_tracker.x1,kcf_tracker.y1), (kcf_tracker.x2,kcf_tracker.y2), (0,255,0), 1)
                        cv2.drawMarker(frame_draw, tuple(kcf_tracker.center), (0,255,0))

                else: # kcf_tracker.enable is False
                    if color_tracker and (zoom_flag is False or zoom.is_zooming is False):
                        if color_tracker.check_interval():
                            mask = {'x1':WIDTH//4, 'y1': HEIGHT//4, 'x2': int(WIDTH*3/4), 'y2': int(HEIGHT*3/4)}
                            color_tracker.update(frame, options = mask, find_by_area=False)
                            if not any(np.isnan(color_tracker.center)):
                                kcf_tracker.x1 = color_tracker.center[0] - selected_width // 2
                                kcf_tracker.x2 = color_tracker.center[0] + selected_width // 2
                                kcf_tracker.y1 = int(color_tracker.center[1] - selected_height / 4)
                                kcf_tracker.y2 = int(color_tracker.center[1] + selected_height * 3 / 4)
                                kcf_tracker.center = ((kcf_tracker.x1 + kcf_tracker.x2) // 2, (kcf_tracker.y1 + kcf_tracker.y2) // 2)
                                kcf_tracker.force_init_flag = True
                                print('[KCF] kcf disabled and color found => force init')
                    elif motion_tracker and (zoom_flag is False or zoom.is_zooming is False) and (motor_flag is False or motor.is_moving is False):
                        if motion_tracker.check_interval():
                            mask = {'x1':0, 'y1': 0, 'x2': WIDTH-1, 'y2': HEIGHT-1}
                            motion_tracker.update(frame, prev_frame, options = mask)
                            if not any(np.isnan(motion_tracker.center)):
                                kcf_tracker.x1 = motion_tracker.x1
                                kcf_tracker.y1 = motion_tracker.y1
                                kcf_tracker.x2 = motion_tracker.x2
                                kcf_tracker.y2 = motion_tracker.y2
                                kcf_tracker.center = motion_tracker.center
                                kcf_tracker.force_init_flag = True
                                print('[KCF] kcf disabled and motion detected => force init')

                            if motor_flag:
                                motor.stop_moving = False
                    else:
                        wide_zoom_flag = False

            elif dlib_tracker:
                if dlib_tracker.force_init_flag is True:
                    print('[DLIB] Force init')
                    dlib_tracker.init(frame)
                    dlib_tracker.force_init_flag = False
                elif dlib_tracker.enable:
                    score, x1, y1, x2, y2 = dlib_tracker.update(frame)
                    # print('size:', (x2-x1)*(y2-y1))
                    if score < 3 or (x2-x1)*(y2-y1) < 144:
                        print('[DLIB] Disabled: score({:.02f}) is too low'.format(score))
                        dlib_tracker.enable = False
                        wide_zoom_flag = True
                    else:
                        wide_zoom_flag = False # just in case
                        dlib_tracker.x1 = x1
                        dlib_tracker.y1 = y1
                        dlib_tracker.x2 = x2
                        dlib_tracker.y2 = y2
                        dlib_tracker.center = ((dlib_tracker.x1 + dlib_tracker.x2) // 2, (dlib_tracker.y1 + dlib_tracker.y2) // 2)

                        dlib_tracker.prev_widths = np.append(dlib_tracker.prev_widths, x2-x1)
                        dlib_tracker.prev_heights = np.append(dlib_tracker.prev_heights, y2-y1)

                        if dlib_tracker.prev_widths.shape[0] > dlib_tracker.PREV_HISTORY_SIZE: # 10
                            dlib_tracker.prev_widths = np.delete(dlib_tracker.prev_widths, (0), axis=0)
                            dlib_tracker.prev_heights = np.delete(dlib_tracker.prev_heights, (0), axis=0)

                        dlib_tracker.mean_width = np.round(np.mean(dlib_tracker.prev_widths)).astype(np.int)
                        dlib_tracker.mean_height = np.round(np.mean(dlib_tracker.prev_heights)).astype(np.int)

                        cv2.rectangle(frame_draw,(dlib_tracker.x1,dlib_tracker.y1), (dlib_tracker.x2,dlib_tracker.y2), (0,255,0), 1)
                        cv2.drawMarker(frame_draw, tuple(dlib_tracker.center), (0,255,0))
                else: # dlib_tracker.enable is False
                    wide_zoom_flag = False

            elif cmt_tracker:
                if cmt_tracker.force_init_flag is True:
                    # print('[CMT]: Force init')
                    cmt_tracker.force_init_flag = False
                    cmt_tracker.init(frame)

                    if cmt_tracker.num_initial_keypoints == 0:
                        print('[CMT] No keypoints found in selection for ({},{}), ({},{})'.format(cmt_tracker.x1, cmt_tracker.y1, cmt_tracker.x2, cmt_tracker.y2))
                        cmt_tracker.force_init_flag = True
                    # else:
                    #     print("[CMT] num_selected_keypoints is {}".format(cmt_tracker.num_initial_keypoints))

                else:
                    cmt_tracker.update(frame)
                    if cmt_tracker.best_effort is False and cmt_tracker.tracked_keypoints.shape[0] < 10: # or cmt_tracker.active_keypoints.shape[0] < 10
                        cmt_tracker.has_result = False

                    # if cmt_tracker.num_of_failure > 0 and cmt_tracker.best_effort is True:
                    #     # print("[CMT] fail count: ", cmt_tracker.num_of_failure)
                    #     cmt_tracker.force_init_flag = True

                    if cmt_tracker.has_result:
                        num_of_tracked_keypoints = len(cmt_tracker.tracked_keypoints)
                        cmt_tracker.cX = int(cmt_tracker.center[0])
                        cmt_tracker.cY = int(cmt_tracker.center[1])

                        box_tl = cmt_tracker.tl
                        box_br = cmt_tracker.br

                        # print("[CMT] {}. Tracked(inlier): {}, Outliers: {}, Votes: {}: Active: {}, Scale: {:02.2f}"
                        #     .format(cmt_tracker.frame_idx, num_of_tracked_keypoints, len(cmt_tracker.outliers), len(cmt_tracker.votes), len(cmt_tracker.active_keypoints), cmt_tracker.scale_estimate))

                        box_center = ((box_tl[0] + box_br[0]) // 2, (box_tl[1] + box_br[1]) // 2)
                        cmt_tracker.box_center = box_center

                        width = box_br[0] - box_tl[0]
                        height = box_br[1] - box_tl[1]

                        (cmt_tracker.x1, cmt_tracker.y1) = box_tl
                        (cmt_tracker.x2, cmt_tracker.y2) = box_br
                        cmt_tracker.area = width * height

                        # util.draw_str(frame, (550, 20), 'Tracking')
                        cv2.rectangle(frame_draw, cmt_tracker.tl, cmt_tracker.br, (0,165,266), 1)
                        cv2.drawMarker(frame_draw, cmt_tracker.box_center, (0,165,255))
                        # cv2.drawMarker(frame, tuple(cmt_tracker.center.astype(np.int16)), (0, 0, 255))

                        # util.draw_keypoints_by_number(cmt_tracker.tracked_keypoints, frame, (0, 0, 255))
                        # util.draw_keypoints_by_number(cmt_tracker.outliers, frame, (255, 0, 0))
                        # util.draw_keypoints(cmt_tracker.tracked_keypoints, frame, (255, 255, 255))
                        # util.draw_keypoints(cmt_tracker.votes[:, :2], frame, (0, 255, 255))
                        # util.draw_keypoints(cmt_tracker.outliers[:, :2], frame, (0, 0, 255))

                        # cv2.drawMarker(frame, (cmt_tracker.cX, cmt_tracker.cY), (255, 255, 255))
                        # cv2.drawMarker(frame, tuple(cmt_tracker.mean_center), (0, 0, 255))
                        # test_tl = box_center[0] - cmt_tracker.mean_width//2, box_center[1] - cmt_tracker.mean_height//2
                        # test_br = box_center[0] + cmt_tracker.mean_width//2, box_center[1] + cmt_tracker.mean_height//2
                        # cv2.rectangle(frame, test_tl, test_br, (255, 255, 255), 1)

                    else: # kcf_tracker.has_result == False
                        pass


            if wide_zoom_flag and zoom_flag and zoom.is_zooming is False and zoom.current_zoom != 1:
                current_zoom = 1
                zoom.zoom_to(current_zoom, dur=2)

            # print('[Debug] motor status:', motor.is_moving)
            if motor_flag and motor.is_moving is not True and motor.stop_moving is False: # and zoom.is_zooming is not True:
                motor.driving_flag = False

                if kcf_tracker and kcf_tracker.enable:
                    cX, cY = kcf_tracker.center
                    motor.driving_flag = True
                elif dlib_tracker and dlib_tracker.enable:
                    cX, cY = dlib_tracker.center
                    motor.driving_flag = True
                elif cmt_tracker and cmt_tracker.has_result:
                    cX, cY = cmt_tracker.box_center
                    motor.driving_flag = True
                else:
                    cX = HALF_WIDTH
                    cY = HALF_HEIGHT

                if motor_flag and motor.driving_flag is True:
                    # cv2.drawMarker(frame, (cX, cY), (0,0,255))
                    center_to_x = cX - HALF_WIDTH
                    center_to_y = HALF_HEIGHT - cY
                    # distance = math.sqrt(center_to_x**2 + center_to_y**2)
                    # print("[MOTOR] Distance from Center: ({}px, {}px)".format(center_to_x, center_to_y))

                    if zoom_flag is False:
                        motor.track(center_to_x, center_to_y)
                    else:
                        motor.track(center_to_x, center_to_y, zoom.current_zoom)


            if (kcf_tracker and kcf_tracker.enable) or (dlib_tracker and dlib_tracker.enable):
                mean_width = kcf_tracker.mean_width if kcf_tracker else dlib_tracker.mean_width
                if autozoom_flag and zoom.is_zooming is False and tracking_is_waiting is False: # and motor.is_moving is False:
                    next_zoom = zoom.find_next_auto_zoom(current_length = mean_width)
                    if next_zoom != zoom.current_zoom:
                        # print("[ZOOM] {} to {}".format(zoom.current_zoom, next_zoom))
                        zoom.zoom_to(next_zoom, dur=2) # hl1sqi dur => 0.1 ?

    if args["display"] is True:
        cv2.line(frame_draw, (HALF_WIDTH, 0), (HALF_WIDTH, WIDTH), (200, 200, 200), 0)
        cv2.line(frame_draw, (0, HALF_HEIGHT), (WIDTH, HALF_HEIGHT), (200, 200, 200), 0)

        if tracking_window['dragging'] == True:
            pt1 = (tracking_window['x1'], tracking_window['y1'])
            pt2 = (tracking_window['x2'], tracking_window['y2'])

            if capture is None:
                capture = np.copy(frame)

            frame_draw = np.copy(capture)
            cv2.rectangle(frame_draw, pt1, pt2, (0, 255, 0,), 1)
            cv2.imshow(title, frame_draw)

        if pause_flag is False:
            if zoom_flag:
                zoom_str = "x{}".format(zoom.current_zoom)
                util.draw_str(frame_draw, (20, 20), zoom_str)

            if motor_flag:
                motor_str = "{:.02f}, {:.02f}".format(motor.sum_of_x_degree, motor.sum_of_y_degree)
                util.draw_str(frame_draw, (20, 340), motor_str)

            if kcf_tracker:
                if tracking_processing_flag and kcf_tracker.enable:
                    kcf_str = "Tracking: on"
                else:
                    kcf_str = "Tracking: off"
                util.draw_str(frame_draw, (WIDTH-120, 20), kcf_str)


            # if color_tracker:
            #     if color_tracker.frame_count > 1:
            #         color_str = "Countdown: {}".format(color_tracker.interval - color_tracker.frame_count)
            #         util.draw_str(frame_draw, (int(WIDTH/2 - 60), 20), color_str)
            #
            # if motion_tracker:
            #     if motion_tracker.frame_count > 1:
            #         motion_str = "Countdown: {}".format(motion_tracker.interval - motion_tracker.frame_count)
            #         util.draw_str(frame_draw, (int(WIDTH/2 - 60), 20), motion_str)

            cv2.imshow(title, frame_draw)
            cv2.moveWindow(title, win_x, win_y)
            if fifo_enable_flag is True:
                fifo.write(frame_draw)

            if upscale_flag is True:
                cv2.imshow('Full frame', frame_full)
                # display_frame_ratio = FRAME_WIDTH/WIDTH # full_frame_to_display_frame_ratio
                #
                # if kcf_tracker and tracking_processing_flag:
                #     x1 = int(kcf_tracker.x1 * display_frame_ratio)
                #     x2 = int(kcf_tracker.x2 * display_frame_ratio)
                #     y0 = int((kcf_tracker.y1*display_frame_ratio + kcf_tracker.y2*display_frame_ratio)/2)
                #     h = int((x2-x1)*9/16)
                #     y1 = y0-h//2
                #     y2 = y0+h//2
                # else:
                #     x1 = 0
                #     y1 = 0
                #     x2 = FRAME_WIDTH-1
                #     y2 = FRAME_HEIGHT-1
                #
                # selected = frame_full[y1:y2, x1:x2]
                # upscale = imutils.resize(selected, width=FRAME_WIDTH)
                # cv2.imshow("Upscale", upscale)

        key = cv2.waitKey(1)
        # print("You pressed {:d} (0x{:x}), 2LSB: {:d} ({:s})".format(key, key, key%2**16, repr(chr(key%256)) if key%256 < 128 else '?'))

        if key == 27 or key == ord('q'):  # ESC or 'q' for 종료
            if gui_flag:
                redis_agent.stop(channel_name)
            break
        elif key == ord(' '): # SPACE for 화면 정지
            pause_flag = not pause_flag
        elif key == ord('s'): # 's' for 트랙킹 중지
            tracking_processing_flag = False
        elif key == ord('u'): # 'u' for 전체 화면 보기
            upscale_flag = not upscale_flag
            if upscale_flag is False:
                cv2.destroyWindow('Full frame')
        elif key == ord('i'): # 'i' for 모터 위치 초기화
            if motor_flag:
                motor.sum_of_x_degree = motor.sum_of_y_degree = 0
                # motor.right_limit = 90
                # motor.left_limit = -90
                # motor.up_limit = 30
                # motor.down_limit = -30
                limit_setup_flag = True
        elif key == ord('w'):
            fifo_enable_flag = not fifo_enable_flag
            if fifo_enable_flag is True:
                print('[FIFO] Enabled')
                fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
                if not os.path.exists('/home/uwtec/Documents/captures'):
                    os.makedirs('/home/uwtec/Documents/captures')

                files = glob.glob('/home/uwtec/Documents/captures/{}-*.mkv'.format(capture_name))
                if len(files) > 0:
                    files.sort()
                    last_file = files[-1]
                    last_num = re.findall(r"[0-9]{4}", last_file)[0]
                    last_num = int(last_num)
                    pic_num = last_num + 1
                else:
                    pic_num = 0

                file_name =  "/home/uwtec/Documents/captures/{}-{:04d}.mkv".format(capture_name, pic_num)
                fifo = cv2.VideoWriter(file_name, fourcc, 30.0, (frame.shape[1], frame.shape[0]))
            else:
                print('[FIFO] Disabled')

        elif key == ord('d'): # 'd' for 모터 각도 보기
            if motor_flag:
                print("[MOTOR] Degree: ({:.02f}, {:.02f})".format(motor.sum_of_x_degree, motor.sum_of_y_degree))

        elif key == ord('l'): # 'l' for 랩 타임 보기
            show_lap_time_flag = not show_lap_time_flag

        elif key%256 == 82: # 'up': 65362 for Ubuntu, 63232 for Mac
            if zoom_flag and zoom.is_zooming is not True:
                next_zoom = zoom.find_next_zoom(dir='in')
                if next_zoom != zoom.current_zoom:
                    zoom.zoom_to(next_zoom, dur=0.1)
        elif key%256 == 84: # 'down': 65364 for Ubuntu, 63233 for Mac
            if zoom_flag and zoom.is_zooming is not True:
                next_zoom = zoom.find_next_zoom(dir='out')
                if next_zoom != zoom.current_zoom:
                    zoom.zoom_to(next_zoom, dur=0.1)
        elif key%256 == 81: # 'left': 65361 for Ubuntu, 63234 for Mac
            if zoom_flag and zoom.is_zooming is not True:
                next_zoom = zoom.find_next_zoom(dir='first')
                if next_zoom != zoom.current_zoom:
                    zoom.zoom_to(next_zoom, dur=0.1)
        elif key%256 == 83: # 'right': 65363 for Ubuntu, 63235 for Mac
            if zoom_flag and zoom.is_zooming is not True:
                next_zoom = zoom.find_next_zoom(dir='last')
                if next_zoom != zoom.current_zoom:
                    zoom.zoom_to(next_zoom, dur=0.1)

        # elif key == ord('f'): # 'f' for Force init
        #     if kcf_tracker:
        #         if color_tracker:
        #             kcf_tracker.enable = False
        #             if zoom:
        #                 wide_zoom_flag = True
        #         elif motion_tracker:
        #             kcf_tracker.enable = False
        #             motion_tracker.frame_count = 75
        #             if zoom:
        #                 wide_zoom_flag = True
        #
        #             if motor:
        #                 motor.stop_moving = True

        # elif key == ord('t'):
        #     print(datetime.datetime.now().time().isoformat())
        #     redis_agent.test(channel_name)

        elif key == ord('t') and cmt_tracker:
            grabbed, test_frame = stream.read()
            test_frame = imutils.resize(test_frame, width=WIDTH)
            gray0 = cv2.cvtColor(test_frame, cv2.COLOR_BGR2GRAY)
            gray0 = cv2.GaussianBlur(gray0, (3, 3), 0)

            for x in range(100, 10, -10):
                detector = cv2.BRISK_create(x, 3, 3.0)
                keypoints = detector.detect(gray0)
                # print("[CMT] {} => {}".format(x, len(keypoints)))
                cmt_detector_threshold = x
                if len(keypoints) >= cmt_tracker.MIN_NUM_OF_KEYPOINTS_FOR_BRISK_THRESHOLD:
                    break
            print("[CMT] BRISK threshold is set to {} with {} keypoints".format(x, len(keypoints)))
            cmt_tracker.detector = detector
            cmt_tracker.descriptor = detector

        if gui_flag:
            if redis_agent.quit:  # ESC or 'q' for 종료
                redis_agent.stop(channel_name)
                break

            elif redis_agent.stop_tracking:
                redis_agent.stop_tracking = False
                tracking_processing_flag = False

            elif redis_agent.zoom_in: # 'up': 65362 for Ubuntu, 63232 for Mac
                redis_agent.zoom_in = False
                if zoom_flag and zoom.is_zooming is not True:
                    next_zoom = zoom.find_next_zoom(dir='in')
                    if next_zoom != zoom.current_zoom:
                        zoom.zoom_to(next_zoom, dur=0.1)
            elif redis_agent.zoom_out: # 'down': 65364 for Ubuntu, 63233 for Mac
                redis_agent.zoom_out = False
                if zoom_flag and zoom.is_zooming is not True:
                    next_zoom = zoom.find_next_zoom(dir='out')
                    if next_zoom != zoom.current_zoom:
                        zoom.zoom_to(next_zoom, dur=0.1)
            elif redis_agent.zoom_x1: # 'left': 65361 for Ubuntu, 63234 for Mac
                redis_agent.zoom_x1 = False
                if zoom_flag and zoom.is_zooming is not True:
                    next_zoom = zoom.find_next_zoom(dir='first')
                    if next_zoom != zoom.current_zoom:
                        zoom.zoom_to(next_zoom, dur=0.1)
            elif redis_agent.autozoom:
                redis_agent.autozoom = False
                if zoom_flag:
                    autozoom_flag = True if redis_agent.autozoom_enable else False
            elif redis_agent.target_scale:
                redis_agent.target_scale = False
                if zoom_flag:
                    zoom.scale = redis_agent.target_scale_value

            elif redis_agent.pause:
                redis_agent.pause = False
                pause_flag = True
            elif redis_agent.play:
                redis_agent.play = False
                pause_flag = False
            elif redis_agent.start_recording:
                redis_agent.start_recording = False
                fifo_enable_flag = True
                fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
                if not os.path.exists('/home/uwtec/Documents/captures'):
                    os.makedirs('/home/uwtec/Documents/captures')

                files = glob.glob('/home/uwtec/Documents/captures/{}-*.mkv'.format(capture_name))
                if len(files) > 0:
                    files.sort()
                    last_file = files[-1]
                    last_num = re.findall(r"[0-9]{4}", last_file)[0]
                    last_num = int(last_num)
                    pic_num = last_num + 1
                else:
                    pic_num = 0

                file_name =  "/home/uwtec/Documents/captures/{}-{:04d}.mkv".format(capture_name, pic_num)
                fifo = cv2.VideoWriter(file_name, fourcc, 30.0, (frame.shape[1], frame.shape[0]))
            elif redis_agent.stop_recording:
                redis_agent.stop_recording = False
                fifo_enable_flag = False

            elif redis_agent.center:
                redis_agent.center = False
                if motor_flag:
                    motor.sum_of_x_degree = motor.sum_of_y_degree = 0
                    # motor.right_limit = 90
                    # motor.left_limit = -90
                    # motor.up_limit = 30
                    # motor.down_limit = -30
                    limit_setup_flag = True

            elif redis_agent.unlock:
                redis_agent.unlock = False
                if motor_flag:
                    motor.sum_of_x_degree = motor.sum_of_y_degree = 0
                    # motor.right_limit = 90
                    # motor.left_limit = -90
                    # motor.up_limit = 30
                    # motor.down_limit = -90
                    # limit_setup_flag = False


        prev_frame = np.copy(frame)

# do a bit of cleanup
cv2.destroyAllWindows()
stream.release()
