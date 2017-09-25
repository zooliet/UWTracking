# -*- coding: utf-8 -*-

from __future__ import print_function
import sys
PY3 = sys.version_info[0] == 3
if PY3:
    xrange = range

import serial
import struct
import datetime
from threading import Timer
import math

class Motor():
    TABLE = [0,  94, 188, 226,  97,  63, 221, 131, 194, 156, 126,  32, 163, 253,  31,  65,
        157, 195,  33, 127, 252, 162,  64,  30,  95,   1, 227, 189,  62,  96, 130, 220,
        35, 125, 159, 193,  66,  28, 254, 160, 225, 191,  93,   3, 128, 222,  60,  98,
        190, 224,   2,  92, 223, 129,  99,  61, 124,  34, 192, 158,  29,  67, 161, 255,
        70,  24, 250, 164,  39, 121, 155, 197, 132, 218,  56, 102, 229, 187,  89,   7,
        219, 133, 103,  57, 186, 228,   6,  88,  25,  71, 165, 251, 120,  38, 196, 154,
        101,  59, 217, 135,   4,  90, 184, 230, 167, 249,  27,  69, 198, 152, 122,  36,
        248, 166,  68,  26, 153, 199,  37, 123,  58, 100, 134, 216,  91,   5, 231, 185,
        140, 210,  48, 110, 237, 179,  81,  15,  78,  16, 242, 172,  47, 113, 147, 205,
        17,  79, 173, 243, 112,  46, 204, 146, 211, 141, 111,  49, 178, 236,  14,  80,
        175, 241,  19,  77, 206, 144, 114,  44, 109,  51, 209, 143,  12,  82, 176, 238,
        50, 108, 142, 208,  83,  13, 239, 177, 240, 174,  76,  18, 145, 207,  45, 115,
        202, 148, 118,  40, 171, 245,  23,  73,   8,  86, 180, 234, 105,  55, 213, 139,
        87,   9, 235, 181,  54, 104, 138, 212, 149, 203,  41, 119, 244, 170,  72,  22,
        233, 183,  85,  11, 136, 214,  52, 106,  43, 117, 151, 201,  74,  20, 246, 168,
        116,  42, 200, 150,  21,  75, 169, 247, 182, 232,  10,  84, 215, 137, 107,  53]

    def __init__(self, dev, baud = 115200, screen_width = 640):
        self.test_count = 0

        if dev:
            self.port = serial.Serial(dev, baud, timeout = 0, parity = serial.PARITY_NONE)
        else:
            self.port = None

        self.is_moving = False

        self.sum_of_x_degree = 0
        self.sum_of_y_degree = 0
        self.DEGREE_PER_PULSE = 0.00048 # 0.00048은 현재 사용 모터와 기어비로 결정되는 펄스 당 회전 각도 (degree)

        self.right_limit = 90 # degree
        self.left_limit = -90
        self.up_limit = 30
        self.down_limit = -30

        self.WIDTH = screen_width # 640x360, 1024x576, 1280x720, 1920x1080
        self.HEIGHT = round(self.WIDTH * 9 / 16)
        self.HALF_WIDTH = round(self.WIDTH / 2)
        self.HALF_HEIGHT = round(self.HEIGHT / 2)

        self.FOVS = {
            1: (62.50, 34.94),
            1.4: (45.20, 25.54), # (45.2, 24.1) x1.38
            1.7: (37.25, 21.16), # (37.5, 20.0) x1.66
            2: (31.25, 17.66), # 2: (31.25, 17.47)
            3: (20.74, 11.71), # (21.02, 11.90)
            4: (15.78, 8.89), # (16.10, 9.12)
            5: (12.56, 7.07), # (12.90, 7.30)
            6: (10.48, 5.89), # (10.94, 5.80)
            7: (9.06, 5.15),  # (9.50, 5.34)
            8: (7.81, 4.42),  # (8.21, 4.65)
            9: (7.00, 3.97),
            10: (6.40, 3.63),
            11: (5.80, 3.26),
            12: (5.20, 2.96),
            13: (4.80, 2.73),
            14: (4.55, 2.55),
            15: (4.25, 2.39),
            16: (3.95, 2.23),
        }

        # fov = [62.50, 35.33]  #[62.50, 34.50] or [68.20, 36.60] or [62.50, 34.94]
        # zooms = [1,1.3,1.7,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
        # self.FOVS = dict((zoom, [round(fov[0]/zoom,2), round(fov[1]/zoom,2)]) for zoom in zooms)
        # print(self.FOVS)

        # 주의: FOVS 변수는 실제는 Half FOVS 값을 가지고 있음
        # self.FOVS = [
        #     (62.5000/2, 34.5000/2), #1
        #     (62.5000/4, 34.5000/4), #2
        #     (62.5000/6, 34.5000/6), #3
        #     (62.5000/8, 34.5000/8), #4
        #     (62.5000/10, 34.5000/10), #5
        #     (62.5000/12, 34.5000/12), #6
        #     (62.5000/14, 34.5000/14), #7
        #     (62.5000/16, 34.5000/16), #8
        #     (62.5000/18, 34.5000/18), #9
        #     (62.5000/20, 34.5000/20), #10
        #     (62.5000/22, 34.5000/22), #11
        #     (62.5000/24, 34.5000/24), #12
        #     (62.5000/26, 34.5000/26), #13
        #     (62.5000/28, 34.5000/28), #14
        #     (62.5000/30, 34.5000/30), #15
        #     (62.5000/32, 34.5000/32), #16
        #     (62.5000/34, 34.5000/34), #17
        #     (62.5000/36, 34.5000/36), #18
        #     (62.5000/38, 34.5000/38), #19
        #     (62.5000/40, 34.5000/40)  #20
        # ]

        self.MANUAL_MOVING_TIME = 1 # 1 sec for default
        self.prev_distance = 0

        self.TRACKING_DURATION = 0.032 # sec

        # full speed: 120000 pulse/sec, half speed: 60000 pulse/sec
        self.SPEEDS_A = {
            1: 60000, 1.4: 42857, 1.7: 35294,
            2: 30000,   3: 20000,
            4: 15000,   5: 12000,   6: 10000,  7: 8571,
            8:  7500,   9: 6667,   10: 6000,  11: 5455,
           12: 5000,   13: 4615,   14: 4286,  15: 4000,
           16: 1500
        }

        self.SPEEDS = {
            1: (64000, 36000),
          1.4: (64000, 36000),
          1.7: (64000, 36000),
            2: (48000, 27000),
            3: (48000, 27000),
            4: (48000, 27000),
            5: (48000, 27000),
            6: (48000, 27000),
            7: (48000, 27000),
            8: (40000, 22500),
            9: (40000, 22500),
           10: (36000, 20250),
           11: (36000, 20250),
           12: (24000, 13500),
           13: (24000, 13500),
           14: (24000, 13500),
           15: (24000, 13500),
           16: (24000, 13500)
        }

        self.STOP_CONDITIONS = {  # self.pixel_to_pulse(64, 36, zoom)
            1: (4340, 2426), # 64x36 => (4340, 2426)
          1.4: (3139, 1774), # 64x36 => (3139, 1774)
          1.7: (2910, 1674), # 64x36 => (2587, 1469), 72x41 => (2910, 1674)
            2: (2441, 1397), # 72x41 => (2441, 1397)
            3: (1620, 926),  # 72x41 => (1620, 926)
            4: (1644, 926),  # 72x41 => (1233, 703), 96x54 => (1644, 926)
            5: (1308, 736),  # 96x54 => (1308, 736)
            6: (1092, 614),  # 96x54 => (1092, 614)
            7: (944, 536),   # 96x54 => (944, 536)
            8: (814, 460),   # 96x54 => (814, 460)
            9: (729, 414),   # 96x54 => (729, 414)
           10: (667, 378),   # 96x54 => (667, 378)
           11: (604, 340),   # 96x54 => (604, 340)
           12: (542, 308),   # 96x54 => (542, 308)
           13: (500, 284),   # 96x54 => (500, 284)
           14: (474, 266),   # 96x54 => (474, 266)
           15: (443, 249),   # 96x54 => (443, 249)
           16: (411, 232)    # 96x54 => (411, 232)
        }

        self.SLOW_CONDITIONS = {  # self.pixel_to_pulse(64, 36, zoom)
            1: (4340, 2426), # 64x36 => (4340, 2426)
          1.4: (3139, 1774), # 64x36 => (3139, 1774)
          1.7: (2910, 1674), # 64x36 => (2587, 1469), 72x41 => (2910, 1674)
            2: (4069, 2317), # 72x41 => (2441, 1397), 120x68 => (4069, 2317)
            3: (2701, 1536), # 72x41 => (1620, 926), 120x68 => (2701, 1536)
            4: (2157, 1543), # 128x68 => (2055, 1166), 160x90 => (2157, 1543)
            5: (2726, 1541), # 160x90 => (2181, 1227), 200x113 => (2726, 1541), 240x135 => (3271, 1841)
            6: (2729, 1534), # 200x113 => (2274, 1284), 240x135 => (2729, 1534)
            7: (2359, 1341), # 200x113 => (1966, 1123), 240x135 => (2359, 1341)
            8: (3051, 1705), # 240x135 => (2034, 1151), 360x200 => (3051, 1705)
            9: (2734, 1532), # 240x135 => (1823, 1034), 360x200 => (2734, 1532)
           10: (2500, 1400), # 360x200 => (2500, 1400)
           11: (2266, 1258), # 360x200 => (2266, 1258)
           12: (2031, 1142), # 360x200 => (2031, 1142)
           13: (2500, 1422), # 360x200 => (1875, 1053), 480x270 => (2500, 1422)
           14: (2370, 1328), # 360x200 => (1777, 984), 480x270 => (2370, 1328)
           15: (2214, 1245), # 360x200 => (1660, 922), 480x270 => (2214, 1245)
           16: (2057, 1161), # 240x135 => (1029, 581), 480x270 => (2057, 1161)
        }

    def has_finished_moving(self, args):
        # args[0] is x_pulse, args[1] is y_pulse
        self.sum_of_x_degree += (args[0] * self.DEGREE_PER_PULSE)
        self.sum_of_y_degree += (args[1] * self.DEGREE_PER_PULSE)
        # print("[MOTOR] End of Moving: {} => {:.2f}".format(args[1], self.sum_of_y_degree)) # hl1sqi

        self.is_moving = False
        # current_time = datetime.datetime.now().time().isoformat()
        # print("[MOTOR] End of Moving: {}".format(current_time)) # hl1sqi

    def crc8_calc(self, data = []):
        crc8 = 0;

        if len(data) == 0:
            test_data = [213, 26, 142, 255, 0, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 64, 66, 15, 0, 0, 55]
            data = test_data[2:-1]

        for x in data:
            crc8 = Motor.TABLE[crc8 ^ x]

        return crc8

    def send_packet(self, buffer):
        crc8 = self.crc8_calc(buffer[2:-1])
        buffer[len(buffer) - 1] = crc8

        bstr = bytes(buffer)
        self.port.write(bstr)

    def move(self, x_pulse = 255, y_pulse = 255, z_pulse = 0, f_pulse = 0,  t = 1, rel = 0xff):
        t = int(t * 1000000) # sec to us
        # print("[MOTOR] Move: ({}, {})pulse in {}us".format(x_pulse, y_pulse, t))

        encoded = list(struct.pack("3i", *[x_pulse, y_pulse, t]))
        buffer = [
            0xd5, 0x1A, 0x8e,
            encoded[0], encoded[1], encoded[2], encoded[3],
            # (x & 0xff) , ((x >> 8) & 0xff), ((x >> 16) & 0xff), ((x >> 24) & 0xff),
            encoded[4], encoded[5], encoded[6], encoded[7],
            # (y & 0xff) , ((y >> 8) & 0xff), ((y >> 16) & 0xff), ((y >> 24) & 0xff),
            0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00,
            encoded[8], encoded[9], encoded[10], encoded[11],
            # (t & 0xff) , ((t >> 8) & 0xff), ((t >> 16) & 0xff), ((t >> 24) & 0xff),
            rel, 0xFF
        ]

        self.send_packet(buffer)

    def deg2rad(self, deg):
        return(deg * math.pi / 180)

    def rad2deg(self, rad):
        return(rad * 180 / math.pi)

    def pixel_to_pulse(self, x_px, y_px, zoom = 1):
        # Logitec
        # x_degree = x_px/320 * 66.59215896/2
        # y_degree = y_px/180 * 40.55232596/2

        hfov = self.FOVS[zoom][0]
        vfov = self.FOVS[zoom][1]

        x_degree = x_px / self.HALF_WIDTH * (hfov/2)
        y_degree = y_px / self.HALF_HEIGHT * (vfov/2)

        x = round(x_degree / self.DEGREE_PER_PULSE)
        y = round(y_degree / self.DEGREE_PER_PULSE)
        z = f = 0

        # print("[MOTOR] ({}, {})px => ({:.4f}, {:.4f})° => ({}, {})pulse".format(x_px, y_px, x_degree, y_degree, x, y))
        return x, y, z, f

    def pulse_to_pixel(self, x_pulse, y_pulse, zoom = 1):
        hfov = self.FOVS[zoom][0]
        vfov = self.FOVS[zoom][1]

        x_degree = x_pulse * self.DEGREE_PER_PULSE
        y_degree = y_pulse * self.DEGREE_PER_PULSE

        x_px = round(x_degree / (hfov/2) * self.HALF_WIDTH)
        y_px = round(y_degree / (vfov/2) * self.HALF_HEIGHT)

        print("[MOTOR] ({}, {})pulse => ({:.4f}, {:.4f})° => ({}, {})px".format(x_pulse, y_pulse, x_degree, y_degree, x_px, y_px))
        return x_px, y_px

    def check_position(self, x_pulse, y_pulse):
        if((self.sum_of_x_degree > self.right_limit and x_pulse > 0) or (self.sum_of_x_degree < self.left_limit and x_pulse < 0)):
            x_pulse = 0
        else:
            sum_of_x_degree = self.sum_of_x_degree + (x_pulse * self.DEGREE_PER_PULSE)
            if sum_of_x_degree > self.right_limit:
                x_pulse = int((self.right_limit - self.sum_of_x_degree) / self.DEGREE_PER_PULSE)
            elif sum_of_x_degree < self.left_limit:
                x_pulse = int((self.left_limit - self.sum_of_x_degree) / self.DEGREE_PER_PULSE)
            # else:
            #     x_pulse = x_pulse

        if (self.sum_of_y_degree > self.up_limit and y_pulse > 0) or (self.sum_of_y_degree < self.down_limit and y_pulse < 0):
            y_pulse = 0
        else:
            sum_of_y_degree = self.sum_of_y_degree + (y_pulse * self.DEGREE_PER_PULSE)
            if sum_of_y_degree > self.up_limit:
                y_pulse = int((self.up_limit - self.sum_of_y_degree) / self.DEGREE_PER_PULSE)
                print("[MOTOR] Current:{:.2f} + {:.2f} = {:.2f}".format(self.sum_of_y_degree, y_pulse, sum_of_y_degree))
            elif sum_of_y_degree < self.down_limit:
                y_pulse = int((self.down_limit - self.sum_of_y_degree) / self.DEGREE_PER_PULSE)
            # else:
            #     y_pulse = y_pulse

        return(x_pulse, y_pulse)

    # 수동 조작에 의한 모터 구동
    def move_to(self, x, y, current_zoom=1):
        if self.port and self.is_moving is False:
            (x_pulse, y_pulse, z_pulse, f_pulse) = self.pixel_to_pulse(x, y, current_zoom)
            (x_pulse, y_pulse) = self.check_position(x_pulse, y_pulse)

            if(x_pulse != 0 or y_pulse != 0):
                self.move(x_pulse = x_pulse, y_pulse = y_pulse, z_pulse = z_pulse, f_pulse = f_pulse, t = self.MANUAL_MOVING_TIME)
                motor_timer = Timer(self.MANUAL_MOVING_TIME, self.has_finished_moving, args = [(x_pulse, y_pulse)])
                motor_timer.start()
                self.is_moving = True
                # current_time = datetime.datetime.now().time().isoformat()
                # print("[MOTOR] Start of Moving: {}".format(current_time)) # hl1sqi

    # 자동 추적에 의한 모터 구동
    def track(self, center, current_zoom=1):
        center_to_x = center[0] - self.HALF_WIDTH
        center_to_y = self.HALF_HEIGHT - center[1]

        if self.port and self.is_moving is False:
            (x_pulse, y_pulse, z_pulse, f_pulse) = self.pixel_to_pulse(center_to_x, center_to_y, current_zoom)
            (x_pulse, y_pulse) = self.check_position(x_pulse, y_pulse)

            if(x_pulse != 0 or y_pulse != 0):
                speed = self.SPEEDS[current_zoom]
                upper_limit = (round(speed[0] * self.TRACKING_DURATION), round(speed[1] * self.TRACKING_DURATION))
                stop_condition = self.STOP_CONDITIONS[current_zoom]
                slow_condition = self.SLOW_CONDITIONS[current_zoom]

                if abs(x_pulse) <= stop_condition[0]:
                    x_pulse = 0
                elif abs(x_pulse) <= slow_condition[0]:
                    x_pulse = round(x_pulse/5)
                elif abs(x_pulse) >= upper_limit[0]:
                    x_pulse = upper_limit[0] if x_pulse >= 0 else -upper_limit[0]
                # else:
                #     x_pulse = x_pluse

                if abs(y_pulse) <= stop_condition[1]:
                    y_pulse = 0
                elif abs(y_pulse) <= slow_condition[1]:
                    y_pulse = round(y_pulse/5)
                elif abs(y_pulse) >= upper_limit[1]:
                    y_pulse = upper_limit[1] if y_pulse >= 0 else -upper_limit[1]
                # else:
                #     y_pulse = y_pulse

                # distance_in_pulse = math.sqrt(x_pulse**2 + y_pulse**2)
                if (abs(x_pulse) > 10 or abs(y_pulse) > 10) and self.test_count < 10:
                    self.move(x_pulse = x_pulse, y_pulse = y_pulse, z_pulse = z_pulse, f_pulse = f_pulse, t = self.TRACKING_DURATION)
                    motor_timer = Timer(self.TRACKING_DURATION * 2, self.has_finished_moving, args = [(x_pulse, y_pulse)])
                    motor_timer.start()
                    self.is_moving = True
                    self.test_count += 0
                    # current_time = datetime.datetime.now().time().isoformat()
                    # print("[MOTOR] ({}, {})px => ({}, {})pulse in {:.0f}ms @{}".format(center_to_x, center_to_y, x_pulse, y_pulse, TIMER*1000, current_time))
        else:
            # print('Request but rejected: ', center_to_x, center_to_y)
            pass
