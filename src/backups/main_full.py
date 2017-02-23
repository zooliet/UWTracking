# import the necessary packages

from __future__ import print_function
import sys
PY3 = sys.version_info[0] == 3
if PY3:
    xrange = range

import cv2
import numpy as np
import imutils

from trackers.color_tracker import ColorTracker
from trackers.kcf_tracker import KCFTracker
from trackers.cmt_tracker import CMTTracker
from trackers.tld_tracker import TLDTracker
from trackers.dlib_tracker import DLIBTracker
from trackers.motion_tracker import MotionTracker
from trackers.kpm_tracker import KPMTracker
from trackers.klt_tracker import KLTTracker
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

# from numpy import isnan
# import redis
# import json

# construct the argument parse and parse the arguments
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--camera", type=int, default=0, help = "camera number")
ap.add_argument("-p", "--path", help = "path to video file")
ap.add_argument("-n", "--num-frames", type=int, default=10000000, help="# of frames to loop over")
ap.add_argument("-d", "--display", action="store_true", help="show display")
ap.add_argument("-s", "--serial", help = "path to serial device")
ap.add_argument("-z", "--zoom", help = "path to zoom control port")
ap.add_argument("-w", "--width", type=int, default=640, help="screen width in px")

ap.add_argument("--kcf", action="store_true", help="Enable KCF tracking")
ap.add_argument("--color", action="store_true", help="Enable color subtracking")
ap.add_argument("--motion", action="store_true", help="Enable Motion subtracking")
ap.add_argument("--autozoom", action="store_true", help="Enable automatic zoom control")

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

if args['path']:
    stream = cv2.VideoCapture(args['path'])
    FRAME_HEIGHT = int(stream.get(4))
    FRAME_WIDTH = int(stream.get(3))
else:
    stream = cv2.VideoCapture(args['camera'])
    stream.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    stream.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

frame_to_display_ratio = FRAME_WIDTH/WIDTH
grabbed, frame_full = stream.read()
frame = imutils.resize(frame_full, width=WIDTH)
prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

pause_flag = False
tracking_processing_flag = False
capture = None
tracking_window = {'x1': -1, 'y1': -1, 'x2': -1, 'y2': -1, 'dragging': False, 'start': False}
show_lap_time_flag = False
upscale_flag = False

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
    motor = Motor(dev = args['serial'], baud = 115200, screen_width = WIDTH)
else:
    motor = None

if args['zoom']:
    zoom = Motor(dev = args['zoom'], baud = 115200, screen_width = WIDTH)
    zoom.zoom_x1()
else:
    zoom = None

if args['color'] is True:
    color_tracker = ColorTracker()
else:
    color_tracker = None

if args['kcf'] is True:
    kcf_tracker = KCFTracker(True, False, True) # hog, fixed_window, multiscale
else:
    kcf_tracker = None

if args['motion'] is True:
    motion_tracker = MotionTracker()
else:
    motion_tracker = None

tic = time.time()
toc = time.time()

while True:
    if pause_flag is False:
        grabbed, frame_full = stream.read()
        if grabbed is not True:
            # print("End of Frame")
            break

        frame = imutils.resize(frame_full, width=WIDTH)
        frame_draw = np.copy(frame)

    # if pause_flag is not True:
        if tracking_window['start'] == True:
            if((tracking_window['x2'] - tracking_window['x1']) > MIN_SELECTION_WIDTH) and ((tracking_window['y2'] - tracking_window['y1']) > MIN_SELECTION_HEIGHT):
                selected_width = tracking_window['x2'] - tracking_window['x1']
                selected_height = tracking_window['y2'] - tracking_window['y1']

                if zoom:
                    selected_width = int(selected_width / zoom.current_zoom)
                    selected_height = int(selected_height / zoom.current_zoom)

                # print("[KCF] User selected width {} and height {}".format(selected_width, selected_height) )

                if color_tracker:
                    if color_tracker.init(frame, options = tracking_window):
                        print('[COLOR] Color Found at {}'.format(color_tracker.center))
                    else:
                        print('[COLOR] Color Not Found around at {}'.format(color_tracker.center))
                    tracking_processing_flag = True # 초기화 결과에 상관없이 tracking 시작

                elif motion_tracker:
                    motion_tracker.motion_count = 0

                if kcf_tracker:
                    kcf_tracker.x1 = tracking_window['x1']
                    kcf_tracker.y1 = tracking_window['y1']
                    kcf_tracker.x2 = tracking_window['x2']
                    kcf_tracker.y2 = tracking_window['y2']

                    #if you use hog feature, there will be a short pause after you draw a first boundingbox, that is due to the use of Numba.
                    kcf_tracker.init(frame)
                    tracking_processing_flag = True # 초기화 결과에 상관없이 tracking 시작



            elif motor and motor.is_moving is not True:
                centerX = (tracking_window['x1'] + tracking_window['x2']) // 2
                centerY = (tracking_window['y1'] + tracking_window['y2']) // 2
                center_to_x = HALF_WIDTH - centerX
                center_to_y = centerY - HALF_HEIGHT
                if zoom is None:
                    motor.move_to(center_to_x, center_to_y)
                elif zoom and zoom.is_zooming is not True:
                    motor.move_to(center_to_x, center_to_y, zoom.current_zoom)

            capture = None
            tracking_window['start'] = False

        if tracking_processing_flag is True:
            if show_lap_time_flag is True: # 'l' key
                current_time = datetime.datetime.now().time().isoformat()
                toc = time.time()
                print("[INFO] Tracking duration: {:04.0f} ms @{}".format(1000*(toc-tic), current_time))
                tic = toc

            if color_tracker:
                if kcf_tracker and kcf_tracker.enable:
                    color_tracker.update(frame, {'x1': kcf_tracker.x1, 'y1':kcf_tracker.y1, 'x2': kcf_tracker.x2, 'y2': kcf_tracker.y2})
                # if kcf_tracker and kcf_tracker.enable is False:
                #     color_tracker.update(frame, {'x1': 0, 'y1': 0, 'x2': WIDTH, 'y2': HEIGHT}, find_by_area = True)
                else:
                    color_tracker.update(frame, {'x1': 0, 'y1': 0, 'x2': WIDTH, 'y2': HEIGHT})

                if color_tracker.consecutive_lost == 0:
                    # cv2.drawMarker(frame_draw, tuple(color_tracker.center), (255, 0, 0), 0)
                    cv2.rectangle(frame_draw, (color_tracker.x1, color_tracker.y1), (color_tracker.x2, color_tracker.y2), (255, 0, 0), 2)

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
                        kcf_tracker.enable = False
                        print('[KCF] Disabled: peak value({:.02f}) is too low'.format(kcf_tracker.peak_value))
                        if color_tracker:
                            color_tracker.consecutive_found = 0 # FOUND_CONDITION을 충족시키는 시간을 벌기위한 조치
                        elif motion_tracker:
                            if motor:
                                motor.stop_moving = True
                                motion_tracker.init(1) # stop_moving 처리 시간이 어차피 필요하니까 인자값을 더 크게?
                            else:
                                motion_tracker.init(1)

                    else:
                        kcf_tracker.x1 = boundingbox[0]
                        kcf_tracker.y1 = boundingbox[1]
                        kcf_tracker.x2 = boundingbox[0] + boundingbox[2]
                        kcf_tracker.y2 = boundingbox[1] + boundingbox[3]
                        kcf_tracker.center = ((kcf_tracker.x1 + kcf_tracker.x2) // 2, (kcf_tracker.y1 + kcf_tracker.y2) // 2)

                    continue_flag = True
                    wide_zoom_flag = False

                    if kcf_tracker.enable == False:
                        wide_zoom_flag = True

                    if color_tracker:
                        if color_tracker.consecutive_found > color_tracker.FOUND_CONDITION:
                            diff_width = abs(color_tracker.center[0] - kcf_tracker.center[0])
                            diff_height = abs(color_tracker.center[1] - kcf_tracker.center[1])
                            if diff_width > boundingbox[2] // 6 or diff_height < boundingbox[3] // 6: # boundingbox[2] == width
                                # print("[KCF] Adjust center with color object")
                                # print("[KCF] Bounding({},{}) vs Mean({},{})".format(boundingbox[2], boundingbox[3], kcf_tracker.mean_width, kcf_tracker.mean_height))
                                kcf_tracker.x1 = color_tracker.center[0] - kcf_tracker.mean_width // 2
                                kcf_tracker.x2 = color_tracker.center[0] + kcf_tracker.mean_width // 2
                                kcf_tracker.y1 = int(color_tracker.center[1] - kcf_tracker.mean_height / 4)
                                kcf_tracker.y2 = int(color_tracker.center[1] + kcf_tracker.mean_height * 3 / 4)
                                kcf_tracker.center = ((kcf_tracker.x1 + kcf_tracker.x2) // 2, (kcf_tracker.y1 + kcf_tracker.y2) // 2)
                                print('[KCF] kcf and color center mismatch => force init')
                                kcf_tracker.force_init_flag = True
                                continue_flag = False

                        elif color_tracker.consecutive_lost >= color_tracker.LOST_CONDITION:
                            kcf_tracker.enable = False
                            kcf_tracker.mean_width = selected_width
                            kcf_tracker.mean_height = selected_height
                            kcf_tracker.prev_widths = np.array([kcf_tracker.mean_width], dtype=np.int16)
                            kcf_tracker.prev_heights = np.array([kcf_tracker.mean_height], dtype=np.int16)
                            continue_flag = False
                            wide_zoom_flag = True

                    elif motion_tracker and motion_tracker.motion_count != 0:
                        continue_flag = False
                        # wide_zoom_flag = True
                    elif kpm_tracker:
                        kpm_tracker.update(frame)
                        # cv2.rectangle(frame_draw, kpm_tracker.tl, kpm_tracker.br, (0,0,255),2)
                        # util.draw_keypoints(kpm_tracker.initial_keypoints, frame_draw, (0, 0, 255))
                        # util.draw_keypoints(kpm_tracker.active_keypoints, frame_draw, (255, 255, 255))

                    #     if kpm_tracker.has_result:
                    #         cv2.rectangle(frame_draw, kpm_tracker.tl, kpm_tracker.br, (0,0,255), 2)

                    if continue_flag is True:
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

                        # if args['autozoom'] and zoom.is_zooming is not True and motor.is_moving is False:
                        #     next_zoom = zoom.find_next_auto_zoom(current_length = kcf_tracker.mean_width)
                        #     if next_zoom != zoom.current_zoom:
                        #         # print("[ZOOM] {} to {}".format(zoom.current_zoom, next_zoom))
                        #         zoom.zoom_to(next_zoom, dur=3)

                    # if wide_zoom_flag is True:
                    #     if zoom and zoom.current_zoom != 1:
                    #         current_zoom = 1
                    #         zoom.zoom_to(current_zoom, dur=3)

                else: # kcf_tracker.enable is False
                    if color_tracker and color_tracker.consecutive_found > color_tracker.FOUND_CONDITION and (zoom is None or zoom.is_zooming is False):
                        kcf_tracker.x1 = color_tracker.center[0] - kcf_tracker.mean_width // 2
                        kcf_tracker.x2 = color_tracker.center[0] + kcf_tracker.mean_width // 2
                        kcf_tracker.y1 = int(color_tracker.center[1] - kcf_tracker.mean_height / 4)
                        kcf_tracker.y2 = int(color_tracker.center[1] + kcf_tracker.mean_height * 3 / 4)
                        kcf_tracker.center = ((kcf_tracker.x1 + kcf_tracker.x2) // 2, (kcf_tracker.y1 + kcf_tracker.y2) // 2)
                        print('[KCF] kcf disabled and color found => force init')
                        kcf_tracker.force_init_flag = True
                    elif motion_tracker and (zoom is None or zoom.is_zooming is False) and (motor is None or motor.is_moving is False):
                        wide_zoom_flag = True
                        if motion_tracker.check_interval():
                            (x1, y1, x2, y2) = motion_tracker.update(frame, prev_frame)
                            if x1 != -1:
                                kcf_tracker.x1 = x1
                                kcf_tracker.y1 = y1
                                kcf_tracker.x2 = x2
                                kcf_tracker.y2 = y2
                                kcf_tracker.force_init_flag = True

                            if motor:
                                motor.stop_moving = False

            if wide_zoom_flag is True:
                if zoom and zoom.current_zoom != 1:
                    current_zoom = 1
                    zoom.zoom_to(current_zoom, dur=3)

            if motor and motor.is_moving is not True and motor.stop_moving is False: # and zoom.is_zooming is not True:
                motor.driving_flag = False

                if kcf_tracker:
                    if kcf_tracker.enable:
                        cX, cY = kcf_tracker.center
                        motor.driving_flag = True
                elif color_tracker and color_tracker.consecutive_lost < color_tracker.FOUND_CONDITION:
                    cX, cY = color_tracker.center
                    motor.driving_flag = True
                elif cmt_tracker and cmt_tracker.has_result:
                    cX, cY = cmt_tracker.box_center
                    motor.driving_flag = True
                else:
                    cX = HALF_WIDTH
                    cY = HALF_HEIGHT

                if args['serial'] and motor.driving_flag is True:
                    # cv2.drawMarker(frame, (cX, cY), (0,0,255))
                    center_to_x = HALF_WIDTH - cX
                    center_to_y = cY - HALF_HEIGHT
                    # print("[MOTOR] Distance from Center: ({}px, {}px)".format(center_to_x, center_to_y))

                    distance = math.sqrt(center_to_x**2 + center_to_y**2)
                    if distance > 4:
                        if zoom is None:
                            motor.track(center_to_x, center_to_y)
                        else:
                            motor.track(center_to_x, center_to_y, zoom.current_zoom)

            if (kcf_tracker and kcf_tracker.enable is True) and args['autozoom'] and zoom.is_zooming is not True: # and motor.is_moving is False:
                next_zoom = zoom.find_next_auto_zoom(current_length = kcf_tracker.mean_width)
                if next_zoom != zoom.current_zoom:
                    # print("[ZOOM] {} to {}".format(zoom.current_zoom, next_zoom))
                    zoom.zoom_to(next_zoom, dur=3)


        cv2.line(frame_draw, (HALF_WIDTH, 0), (HALF_WIDTH, WIDTH), (200, 200, 200), 0)
        cv2.line(frame_draw, (0, HALF_HEIGHT), (WIDTH, HALF_HEIGHT), (200, 200, 200), 0)


    if args["display"] is True:
        if tracking_window['dragging'] == True:
            pt1 = (tracking_window['x1'], tracking_window['y1'])
            pt2 = (tracking_window['x2'], tracking_window['y2'])

            if capture is None:
                capture = np.copy(frame)

            frame_draw = np.copy(capture)
            cv2.rectangle(frame_draw, pt1, pt2, (0, 255, 0,), 1)
            cv2.imshow("Tracking", frame_draw)

        # pause가 아닌 상태에서 Tracking window 보이기(당연),
        # 그런데 pause 일때 굳이 동작 않도록 처리한 이유는? => pause 일때 마우스 조작이 일어나는 경우에 대처하기 위해, 즉, 다른곳에서 윈도우 처리
        if pause_flag is False:
            zoom_str = "x{}".format(zoom.current_zoom)
            util.draw_str(frame_draw, (20, 20), zoom_str)
            kcf_str = "Processing: {}, Enable: {}".format(tracking_processing_flag, kcf_tracker.enable)
            util.draw_str(frame_draw, (WIDTH-300, 20), kcf_str)

            color_str = "Lost: {} and Found: {}".format(color_tracker.consecutive_lost, color_tracker.consecutive_found)
            util.draw_str(frame_draw, (WIDTH//2-100, 20), color_str)

            cv2.imshow("Tracking", frame_draw)

        if upscale_flag is True:
            if kcf_tracker and tracking_processing_flag:
                x1 = int(kcf_tracker.x1 * frame_to_display_ratio)
                x2 = int(kcf_tracker.x2 * frame_to_display_ratio)
                y0 = int((kcf_tracker.y1*frame_to_display_ratio + kcf_tracker.y2*frame_to_display_ratio)/2)
                h = int((x2-x1)*9/16)
                y1 = y0-h//2
                y2 = y0+h//2
            else:
                x1 = 0
                y1 = 0
                x2 = FRAME_WIDTH-1
                y2 = FRAME_HEIGHT-1

            selected = frame_full[y1:y2, x1:x2]
            upscale = imutils.resize(selected, width=FRAME_WIDTH)
            cv2.imshow("Upscale", upscale)

        key = cv2.waitKey(1)
        if key == 27 or key == ord('q'):
            break
        elif key == ord(' '):
            pause_flag = not pause_flag
        elif key == ord('s'):
            tracking_processing_flag = False
        elif key == ord('u'):
            upscale_flag = not upscale_flag
            if upscale_flag is False:
                cv2.destroyWindow('Upscale')
        elif key == ord('f'):
            if kcf_tracker:
                # print("[DEBUG] kcf enabled: {}, zoom: {}, color: {}/{}".format(kcf_tracker.enable, zoom.is_moving, color_tracker.consecutive_lost, color_tracker.consecutive_found))
                if motion_tracker:
                    # wide_zoom_flag = True
                    if motor:
                        kcf_tracker.enable = False
                        motor.stop_moving = True
                        motion_tracker.init(5) # stop_moving 처리 시간이 어차피 필요하니까 인자값을 더 크게?
                    else:
                        motion_tracker.init(5)
                        kcf_tracker.enable = False
                        # 어차피 모터 제어가 안되면 기다리지 말고 바로 수행할 수도
                        # (x1, y1, x2, y2) = motion_tracker.update(frame, prev_frame)
                        # if x1 != -1:
                        #     kcf_tracker.x1 = x1
                        #     kcf_tracker.y1 = y1
                        #     kcf_tracker.x2 = x2
                        #     kcf_tracker.y2 = y2
                        #     kcf_tracker.force_init_flag = True
                # elif kpm_tracker:
                #     kpm_tracker.update(frame)
                #     if kpm_tracker.has_result:
                #         print("[KPM] has result: {}, {}".format(kpm_tracker.tl, kpm_tracker.br))
                #         (kcf_tracker.x1, kcf_tracker.y1) = kpm_tracker.tl
                #         (kcf_tracker.x2, kcf_tracker.y2) = kpm_tracker.br
                #         kcf_tracker.force_init_flag = True
                #         # cv2.rectangle(frame_draw, kpm_tracker.tl, kpm_tracker.br, (0,0,255), 2)
                #
                #     else:
                #         print("[KPM] No result")

            # elif dlib_tracker:
            #     dlib_tracker.match(frame)
            #     dlib_tracker.find_motion(frame, prev_frame)

        elif key == 65362: # 'up', 63232 for Mac
            if zoom and zoom.is_zooming is not True:
                next_zoom = zoom.find_next_zoom(dir='in')
                if next_zoom != zoom.current_zoom:
                    zoom.zoom_to(next_zoom, dur=0.1)
        elif key == 65364: # 'down', 63233 for Mac
            if zoom and zoom.is_zooming is not True:
                next_zoom = zoom.find_next_zoom(dir='out')
                if next_zoom != zoom.current_zoom:
                    zoom.zoom_to(next_zoom, dur=0.1)
        elif key == 65361: # 'left', 63234 for Mac
            if zoom and zoom.is_zooming is not True:
                next_zoom = zoom.find_next_zoom(dir='first')
                if next_zoom != zoom.current_zoom:
                    zoom.zoom_to(next_zoom, dur=0.1)
        elif key == 65363: # 'right', 63235 for Mac
            if zoom and zoom.is_zooming is not True:
                next_zoom = zoom.find_next_zoom(dir='last')
                if next_zoom != zoom.current_zoom:
                    zoom.zoom_to(next_zoom, dur=0.1)
        elif key == ord('d'):
            print("[MOTOR] Degree: ({:.02f}, {:.02f})".format(motor.sum_of_x_degree, motor.sum_of_y_degree))
        elif key == ord('t') and (args['cmt_alone'] or args['cmt']) is True:
            grabbed, frame_full = stream.read()
            frame = imutils.resize(frame_full, width=WIDTH)
            gray0 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray0 = cv2.GaussianBlur(gray0, (3, 3), 0)

            for x in range(10, 500, 10):
                detector = cv2.BRISK_create(x, 3, 3.0)
                keypoints = detector.detect(gray0)
                cmt_detector_threshold = x
                if len(keypoints) < cmt_tracker.MIN_NUM_OF_KEYPOINTS_FOR_BRISK_THRESHOLD:
                    break
            print("[CMT] BRISK threshold is set to {} with {} keypoints".format(x, len(keypoints)))
            cmt_tracker.detector = detector
            cmt_tracker.descriptor = detector
        elif key == ord('l'):
            show_lap_time_flag = not show_lap_time_flag
        elif key == ord('i'):
            if args['serial']:
                motor.sum_of_x_degree = motor.sum_of_y_degree = 0

        prev_frame = np.copy(frame)
# do a bit of cleanup
cv2.destroyAllWindows()
stream.release()
