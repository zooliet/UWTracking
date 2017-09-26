# -*- coding: utf-8 -*-

from __future__ import print_function
import sys
PY3 = sys.version_info[0] == 3
if PY3:
    xrange = range

import os
import glob
import re
import math
import time
import datetime
from threading import Timer
import redis
import json

import cv2
import numpy as np
import imutils

from trackers.kcf_tracker import KCFTracker

from motor import Motor
from zoom import Zoom
from utils import util
from redis_agent import RedisAgent

# from trackers.kcf.kcf_tracker import KCFTracker
# from trackers.color.color_tracker import ColorTracker
# from trackers.motion.motion_tracker import MotionTracker
# from trackers.dlib.dlib_tracker import DLIBTracker
# from trackers.cmt.cmt_tracker import CMTTracker
# from utils import common

################################################################################

cfg = {
    'FRAME_WIDTH': 1920,
    'FRAME_HEIGHT': 1080,
}

from utils import command_parsing as cli
args = cli.parsing()

# WIDTH, HEIGHT, HALF_WIDTH, HALF_HEIGHT, MIN_SELECTION_WIDTH, MIN_SELECTION_HEIGHT
cli.setWidth( args.get('width'), cfg )

# TITLE, WIN_X, WIN_Y, CHANNEL_NAME, CAPTURE_NAME
cli.setView( args.get('view'), cfg )
# print(cfg)
################################################################################

if args['path']:
    stream = cv2.VideoCapture(args['path'])
    cfg['FRAME_HEIGHT'] = int(stream.get(4))
    cfg['FRAME_WIDTH'] = int(stream.get(3))
else:
    if args['camera'].isdigit():
        stream = cv2.VideoCapture(int(args['camera']))
    else:
        stream = cv2.VideoCapture(args['camera'])
    stream.set(cv2.CAP_PROP_FRAME_WIDTH, cfg['FRAME_WIDTH'])
    stream.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg['FRAME_HEIGHT'])

################################################################################

motor = Motor(dev = args['motor'], baud = 115200, screen_width = cfg['WIDTH'])
zoom = Zoom(dev = args['zoom'], baud = 115200, screen_width = cfg['WIDTH'])
redis_agent = RedisAgent(redis.Redis(), [cfg['CHANNEL_NAME']])
redis_agent.start()

################################################################################

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

cv2.namedWindow(cfg['TITLE'])
cv2.moveWindow(cfg['TITLE'], cfg['WIN_X'], cfg['WIN_Y'])
tracking_window = {'x1': -1, 'y1': -1, 'x2': -1, 'y2': -1, 'dragging': False, 'start': False}
cv2.setMouseCallback(cfg['TITLE'], onmouse, tracking_window)

################################################################################

# Read the first frame
grabbed, frame_full = stream.read()
frame = imutils.resize(frame_full, width=640)
prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

grabbed, frame_full = stream.read()
frame = imutils.resize(frame_full, width=640)
prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

################################################################################

if args['kcf'] is True:
    kcf_tracker = KCFTracker(True, False, True) # hog, fixed_window, multiscale
else:
    kcf_tracker = None

################################################################################

capture = None
pause_flag = False
tracking_processing_flag = False
upscale_flag = False
fifo_enable_flag = False
show_lap_time_flag = False
request_to_track_flag = False
autozoom_flag = args['autozoom']

tic = time.time()
toc = time.time()

################################################################################

while True:
    if pause_flag is False:
        grabbed, frame_full = stream.read()
        if grabbed is not True:
            # print("End of Frame")
            break

        frame = imutils.resize(frame_full, width=cfg['WIDTH'])
        frame_draw = np.copy(frame)

        if tracking_window['start'] == True:
            # w = tracking_window['x2'] - tracking_window['x1']
            # h = tracking_window['y2'] - tracking_window['y1']
            # print("Area: {} = {}x{}".format(w*h, w, h))
            if((tracking_window['x2'] - tracking_window['x1']) > cfg['MIN_SELECTION_WIDTH']) and \
               ((tracking_window['y2'] - tracking_window['y1']) > cfg['MIN_SELECTION_HEIGHT']):
                selected_width = tracking_window['x2'] - tracking_window['x1']
                selected_height = tracking_window['y2'] - tracking_window['y1']

                if kcf_tracker:
                    kcf_tracker.x1 = tracking_window['x1']
                    kcf_tracker.y1 = tracking_window['y1']
                    kcf_tracker.x2 = tracking_window['x2']
                    kcf_tracker.y2 = tracking_window['y2']

                    #if you use hog feature, there will be a short pause after you draw a first boundingbox, that is due to the use of Numba.
                    kcf_tracker.init(frame)

                    tracking_processing_flag = True
                    # tracking_is_waiting = True
                    # tracking_timer = Timer(3, tracking_processing_delay, args = [False])
                    # tracking_timer.start()
            else: # tracking_processing_flag is False:
                centerX = round((tracking_window['x1'] + tracking_window['x2']) / 2)
                centerY = round((tracking_window['y1'] + tracking_window['y2']) / 2)
                center_to_x = centerX - cfg['HALF_WIDTH']
                center_to_y = cfg['HALF_HEIGHT'] - centerY
                # motor.move_to(center_to_x, center_to_y)
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
                    if kcf_tracker.peak_value < 0.1: # 0.25:
                        print('[KCF] Disabled: peak value({:.02f}) is too low'.format(kcf_tracker.peak_value))
                        kcf_tracker.enable = False
                        zoom.zoom_to(1)

                    else:
                        kcf_tracker.x1 = boundingbox[0]
                        kcf_tracker.y1 = boundingbox[1]
                        kcf_tracker.x2 = boundingbox[0] + boundingbox[2]
                        kcf_tracker.y2 = boundingbox[1] + boundingbox[3]
                        kcf_tracker.center = (round((kcf_tracker.x1 + kcf_tracker.x2) / 2), round((kcf_tracker.y1 + kcf_tracker.y2) / 2))

                        kcf_tracker.prev_widths = np.append(kcf_tracker.prev_widths, boundingbox[2])
                        kcf_tracker.prev_heights = np.append(kcf_tracker.prev_heights, boundingbox[3])

                        if kcf_tracker.prev_widths.shape[0] > kcf_tracker.PREV_HISTORY_SIZE: # 10
                            kcf_tracker.prev_widths = np.delete(kcf_tracker.prev_widths, (0), axis=0)
                            kcf_tracker.prev_heights = np.delete(kcf_tracker.prev_heights, (0), axis=0)

                        kcf_tracker.mean_width = np.round(np.mean(kcf_tracker.prev_widths)).astype(np.int)
                        kcf_tracker.mean_height = np.round(np.mean(kcf_tracker.prev_heights)).astype(np.int)

                        # str = "{}x{}({}x{})".format(kcf_tracker.mean_width, kcf_tracker.mean_height, selected_width, selected_height)
                        # util.draw_str(frame_draw, (20, 20), str)
                        cv2.rectangle(frame_draw,(kcf_tracker.x1,kcf_tracker.y1), (kcf_tracker.x2,kcf_tracker.y2), (0,255,0), 1)
                        cv2.drawMarker(frame_draw, tuple(kcf_tracker.center), (0,255,0))
                        request_to_track_flag = True

                else:
                    pass

            if request_to_track_flag is True:
                request_to_track_flag = False
                if kcf_tracker.enable:
                    motor.track(kcf_tracker.center, zoom.current_zoom)
                    if autozoom_flag:
                        zoom.autozoom(width = kcf_tracker.mean_width, height = kcf_tracker.mean_height)

    if tracking_window['dragging'] == True:
        pt1 = (tracking_window['x1'], tracking_window['y1'])
        pt2 = (tracking_window['x2'], tracking_window['y2'])

        if capture is None:
            capture = np.copy(frame)

        frame_draw = np.copy(capture)
        cv2.rectangle(frame_draw, pt1, pt2, (0, 255, 0,), 1)
        cv2.imshow(cfg['TITLE'], frame_draw)

    cv2.line(frame_draw, (cfg['HALF_WIDTH'], 0), (cfg['HALF_WIDTH'], cfg['HEIGHT']), (200, 200, 200), 0)
    cv2.line(frame_draw, (0, cfg['HALF_HEIGHT']), (cfg['WIDTH'], cfg['HALF_HEIGHT']), (200, 200, 200), 0)

    # cv2.line(frame_draw, (int(cfg['WIDTH']*1/4), 0), (int(cfg['WIDTH']*1/4), cfg['HEIGHT']), (0, 0, 200), 0)
    # cv2.line(frame_draw, (int(cfg['WIDTH']*3/4), 0), (int(cfg['WIDTH']*3/4), cfg['HEIGHT']), (0, 0, 200), 0)
    # cv2.line(frame_draw, (0, int(cfg['HEIGHT']*1/4)), (cfg['WIDTH'], int(cfg['HEIGHT']*1/4)), (0, 0, 200), 0)
    # cv2.line(frame_draw, (0, int(cfg['HEIGHT']*3/4)), (cfg['WIDTH'], int(cfg['HEIGHT']*3/4)), (0, 0, 200), 0)

    if pause_flag is False:
        zoom_str = "x{}".format(zoom.current_zoom)
        if autozoom_flag:
            zoom_str += " (Auto)"
            util.draw_str(frame_draw, (cfg['WIDTH']-100, cfg['HEIGHT'] - 20), zoom_str, (0, 0, 200))
        else:
            util.draw_str(frame_draw, (cfg['WIDTH']-50, cfg['HEIGHT'] - 20), zoom_str, (0, 0, 200))

        position_str = "{:.02f}, {:.02f}".format(motor.sum_of_x_degree, motor.sum_of_y_degree)
        util.draw_str(frame_draw, (10, cfg['HEIGHT'] - 30), position_str, (0, 0, 200))

        if tracking_processing_flag and kcf_tracker.enable:
            kcf_str = "Tracking: on"
        else:
            kcf_str = "Tracking: off"
        util.draw_str(frame_draw, (cfg['WIDTH']-120, 20), kcf_str, (0, 0, 200))

        cv2.imshow(cfg['TITLE'], frame_draw)
        cv2.moveWindow(cfg['TITLE'], cfg['WIN_X'], cfg['WIN_Y'])

        if fifo_enable_flag is True:
            fifo.write(frame_draw)

        if upscale_flag is True:
            cv2.imshow('Full frame', frame_full)

    key = cv2.waitKey(1)
    # print("You pressed {:d} (0x{:x}), 2LSB: {:d} ({:s})".format(key, key, key%2**16, repr(chr(key%256)) if key%256 < 128 else '?'))

    if key == 27 or key == ord('q') or redis_agent.quit:  # ESC or 'q' for 종료
        redis_agent.stop(cfg['CHANNEL_NAME'])
        break

    elif key == ord(' '): # SPACE for 화면 정지
        pause_flag = not pause_flag
    elif redis_agent.pause:
        redis_agent.pause = False
        pause_flag = True
    elif redis_agent.play:
        redis_agent.play = False
        pause_flag = False

    elif key == ord('s') or redis_agent.stop_tracking: # 's' for 트랙킹 중지
        redis_agent.stop_tracking = False
        if tracking_processing_flag == True:
            tracking_processing_flag = False

    elif key == ord('u'): # 'u' for 전체 화면 보기
        upscale_flag = not upscale_flag
        if upscale_flag is False:
            cv2.destroyWindow('Full frame')

    elif key == ord('i') or redis_agent.center: # 'i' for 모터 위치 초기화
        redis_agent.center = False
        motor.sum_of_x_degree = motor.sum_of_y_degree = 0
        motor.test_count = 0

    elif key == ord('w') or redis_agent.start_recording: # 'w' for 파일 쓰기
        if redis_agent.start_recording:
            redis_agent.start_recording = False
            fifo_enable_flag = True
        else:
            fifo_enable_flag = not fifo_enable_flag

        if fifo_enable_flag is True:
            # print('[FIFO] Enabled')
            fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
            if not os.path.exists('/home/uwtec/Documents/captures'):
                os.makedirs('/home/uwtec/Documents/captures')

            files = glob.glob('/home/uwtec/Documents/captures/{}-*.mkv'.format(cfg['CAPTURE_NAME']))
            if len(files) > 0:
                files.sort()
                last_file = files[-1]
                last_num = re.findall(r"[0-9]{4}", last_file)[0]
                last_num = int(last_num)
                pic_num = last_num + 1
            else:
                pic_num = 0

            file_name =  "/home/uwtec/Documents/captures/{}-{:04d}.mkv".format(cfg['CAPTURE_NAME'], pic_num)
            fifo = cv2.VideoWriter(file_name, fourcc, 30.0, (frame.shape[1], frame.shape[0]))
        else:
            # print('[FIFO] Disabled')
            pass

    elif redis_agent.stop_recording:
        redis_agent.stop_recording = False
        fifo_enable_flag = False

    elif key%256 == 81 or redis_agent.zoom_x1: # 'left': 65361 for Ubuntu, 63234 for Mac
        redis_agent.zoom_x1 = False
        zoom.zoom_to(1)
    elif key%256 == 83 or redis_agent.zoom_x16: # 'right': 65363 for Ubuntu, 63235 for Mac
        redis_agent.zoom_x16 = False
        zoom.zoom_to(16)
    elif key%256 == 82 or redis_agent.zoom_in: # 'up': 65362 for Ubuntu, 63232 for Mac
        redis_agent.zoom_in = False
        zoom.zoom_in()
    elif key%256 == 84 or redis_agent.zoom_out: # 'down': 65364 for Ubuntu, 63233 for Mac
        redis_agent.zoom_out = False
        zoom.zoom_out()

    elif key == ord('a'):
        autozoom_flag = not autozoom_flag
    elif redis_agent.autozoom:
        redis_agent.autozoom = False
        autozoom_flag = True if redis_agent.autozoom_enable else False
    elif redis_agent.auto_scale:
        redis_agent.auto_scale = False
        zoom.auto_scale = redis_agent.auto_scale_value

    elif key == ord('t'):
        print(datetime.datetime.now().time().isoformat())
        redis_agent.test(cfg['CHANNEL_NAME'])
    elif key == ord('l'): # 'l' for 랩 타임 보기
        show_lap_time_flag = not show_lap_time_flag

    prev_frame = np.copy(frame)

# do a bit of cleanup
cv2.destroyAllWindows()
stream.release()
