import struct
import serial
import time
from threading import Timer
import math
import datetime
import time

class Motor:
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

    def __init__(self, dev = '/dev/ttyUSB0', baud = 115200, screen_width = 640):
        self.port = serial.Serial(dev, baud, timeout = 0, parity = serial.PARITY_NONE)
        self.sum_of_x_degree = 0
        self.sum_of_y_degree = 0

        self.is_moving = False
        self.is_zooming = False
        self.stop_moving = False
        # self.zoom_to_preset = {1: 1, 2: 2, 4: 3, 8: 4, 16: 5}
        self.zoom_to_preset = {1: 1, 2: 2, 4: 3, 8: 4, 14: 5}
        self.available_zooms = sorted(self.zoom_to_preset.keys())
        # self.available_zooms = [1,2,4,8,16]
        self.current_zoom = 1

        self.WIDTH = screen_width # 640x360, 1024x576, 1280x720, 1920x1080
        self.HEIGHT = int(self.WIDTH * 9 / 16)
        self.HALF_WIDTH = self.WIDTH // 2
        self.HALF_HEIGHT = self.HEIGHT // 2
        self.scale = 0.3

        self.right_limit = 90
        self.left_limit = -90
        self.up_limit = 30 # 90
        self.down_limit = -90

        self.DEGREE_PER_PULSE = 0.00048 # 0.00048은 현재 사용 모터와 기어비로 결정되는 펄스 당 회전 각도 (degree)
        self.MANUAL_MOVING_TIME = 1 # 1 sec for default

        self.prev_distance = 0
        self.tic = time.time()
        toc = time.time()

        # 주의: FOVS 변수는 실제는 Half FOVS 값을 가지고 있음
        self.FOVS = [(62.5000/2, 34.5000/2), #1
                    (62.5000/4, 34.5000/4), #2
                    (0, 0),
                    (62.5000/8, 34.5000/8), #4
                    (0, 0),
                    (0, 0),
                    (0, 0),
                    (62.5000/16, 34.5000/16), #8
                    (0, 0),
                    (0, 0),
                    (0, 0),
                    (62.5000/24, 34.5000/24), #12
                    (0, 0),
                    (62.5000/28, 34.5000/28), #12
                    (0, 0),
                    (62.5000/32, 34.5000/32), #16
                    (0, 0),
                    (0, 0),
                    (0, 0),
                    (62.5000/40, 34.5000/40), ] #20

    def has_finished_moving(self, args):
        self.is_moving = False
        # current_time = datetime.datetime.now().time().isoformat()
        # print("[MOTOR] End of Moving: {}".format(current_time)) # hl1sqi

    def send_packet(self, buffer):
        crc8 = self.crc8_calc(buffer[2:-1])
        buffer[len(buffer) - 1] = crc8

        bstr = bytes(buffer)

        self.port.write(bstr)

    def crc8_calc(self, data = []):
        crc8 = 0;

        if len(data) == 0:
            test_data = [213, 26, 142, 255, 0, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 64, 66, 15, 0, 0, 55]
            data = test_data[2:-1]

        for x in data:
            crc8 = Motor.TABLE[crc8 ^ x]

        return crc8

    def move(self, x = 255, y = 255, z = 0, f = 0,  t = 1, rel = 0xff):
        t = int(t * 1000000) # sec to us
        # print("[MOTOR] Move: x: {} y: {} t: {} rel: {}".format(x, y, t, rel))

        sum_of_x_degree = self.sum_of_x_degree + (x * self.DEGREE_PER_PULSE)
        sum_of_y_degree = self.sum_of_y_degree + (y * self.DEGREE_PER_PULSE)

        if sum_of_x_degree > self.right_limit:
            x = int((self.right_limit - self.sum_of_x_degree) / self.DEGREE_PER_PULSE)
        elif sum_of_x_degree < self.left_limit:
            x = int((self.left_limit - self.sum_of_x_degree) / self.DEGREE_PER_PULSE)

        if sum_of_y_degree > self.up_limit:
            y = int((self.up_limit - self.sum_of_y_degree) / self.DEGREE_PER_PULSE)
        elif sum_of_y_degree < self.down_limit:
            y = int((self.down_limit - self.sum_of_y_degree) / self.DEGREE_PER_PULSE)

        # if (self.sum_of_x_degree > self.right_limit and x > 0) or (self.sum_of_x_degree < self.left_limit and x < 0):
        #     x = 0
        # else:
        self.sum_of_x_degree += (x * self.DEGREE_PER_PULSE)

        # if (self.sum_of_y_degree > self.up_limit and y > 0) or (self.sum_of_y_degree < self.down_limit and y < 0):
        #     y = 0
        # else:
        self.sum_of_y_degree += (y * self.DEGREE_PER_PULSE)

        encoded = list(struct.pack("3i", *[x, y, t]))

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

    def pulse_to_pixel(self, x_pulse, y_pulse, zoom = 1):
        x_degree = x_pulse * self.DEGREE_PER_PULSE
        y_degree = y_pulse * self.DEGREE_PER_PULSE

        x_px = round(x_degree * self.HALF_WIDTH / self.FOVS[zoom-1][0])
        y_px = round(y_degree * self.HALF_HEIGHT / self.FOVS[zoom-1][1])

        return x_px, y_px

    def pixel_to_pulse(self, x_px, y_px, zoom = 1, limit = False):
        # Logitec
        # x_degree = x_px/320 * 66.59215896/2
        # y_degree = y_px/180 * 40.55232596/2

        # UWTec
        # x_degree = x_px/self.HALF_WIDTH * 62.0000/2
        # y_degree = y_px/self.HALF_HEIGHT * 34.5000/2

        x_degree = x_px/self.HALF_WIDTH * self.FOVS[zoom-1][0]
        y_degree = y_px/self.HALF_HEIGHT * self.FOVS[zoom-1][1]

        x = round(x_degree/self.DEGREE_PER_PULSE)
        y = round(y_degree/self.DEGREE_PER_PULSE)
        z = f = 0

        # print("[MOTOR] ({}px, {}px) => ({:.4f}°, {:.4f}°) => ({}, {}) pulse".format(x_px, y_px, x_degree, y_degree, x, y))
        return x, y, z, f

    def move_to(self, x, y, current_zoom=1):
        (x_to, y_to, z_to, f_to) = self.pixel_to_pulse(x, y, current_zoom, limit = False)
        self.move(x = x_to, y = y_to, z = z_to, f = f_to, t = self.MANUAL_MOVING_TIME)

        # hl1sqi
        # motor_timer = Timer(self.MANUAL_MOVING_TIME * 2, self.has_finished_moving, args = [False])
        motor_timer = Timer(self.MANUAL_MOVING_TIME, self.has_finished_moving, args = [False])

        motor_timer.start()
        self.is_moving = True
        # current_time = datetime.datetime.now().time().isoformat()
        # print("[MOTOR] Start of Moving: {}".format(current_time)) # hl1sqi

    def track(self, center_to_x, center_to_y, current_zoom=1):
        toc = time.time()
        lap = (toc - self.tic) * 1000
        # print("{:04.0f} ms".format(lap))
        self.tic = toc

        # Case 1
        # SPEED = 12000 # full speed: 120000, half speed: 600000
        # SPEEDS = list(map(lambda idx: int(30000 * self.FOVS[idx-1][0]/self.FOVS[0][0]), list(range(0,21))))
        # [30000, 30000, 15000, 0, 7500, 0, 0, 0, 3750, 0, 0, 0, 2500, 0, 0, 0, 1875, 0, 0, 0, 1500]
        SPEEDS = [0, 30000, 30000, 0, 30000, 0, 0, 0, 30000, 0, 0, 0, 0, 0, 15000, 0, 15000, 0, 0, 0, 0]
        SPEED = SPEEDS[current_zoom]
        # SPEED = 30000

        (x_pulse, y_pulse, z_pulse, f_pulse) = self.pixel_to_pulse(center_to_x, center_to_y, current_zoom, limit = True)
        d_pulse = max(abs(x_pulse), abs(y_pulse))
        t_sec = d_pulse / SPEED # 소요 시간
        # print("[MOTOR] ({}px, {}px) => ({}, {}) pulse in {:.0f} ms @{:04.0f}".format(center_to_x, center_to_y, x_pulse, y_pulse, t_sec*1000, lap))


        MIN_DISTANCES = [0, 10, 20, 0, 30, 0, 0, 0, 60, 0, 0, 0, 0, 0, 60, 0, 60, 0, 0, 0, 0]
        MIN_DISTANCE = MIN_DISTANCES[current_zoom]

        MIN_MOVING_DISTANCES = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]
        MIN_MOVING_DISTANCE = MIN_MOVING_DISTANCES[current_zoom]

        MOVING_TIME = 0.017 # 0.1 for 100 ms

        if t_sec >= MOVING_TIME: # and t_sec > 0:
            x_pulse = int(x_pulse * (MOVING_TIME / t_sec))
            y_pulse = int(y_pulse * (MOVING_TIME / t_sec))
            x_px, y_px = self.pulse_to_pixel(x_pulse, y_pulse, current_zoom)
            t_sec = MOVING_TIME

            distance = math.sqrt(center_to_x**2 + center_to_y**2)
            # print("[MOTOR] Distance from Center: ({}px, {}px) => {:.0f}".format(center_to_x, center_to_y, distance))

            if abs(distance - self.prev_distance) > MIN_MOVING_DISTANCE and distance > MIN_DISTANCE:
                # print("[MOTOR: DO] ({}px, {}px) => ({:.0f}px, {:.0f}px) => ({}, {}) pulse in {:.0f} ms".format(center_to_x, center_to_y, x_px, y_px, x_pulse, y_pulse, t_sec*1000))
                self.prev_distance = distance
                self.move(x = x_pulse, y = y_pulse, z = z_pulse, f = f_pulse, t = t_sec)
            # else:
            #     print("[MOTOR: DON'T] ({}px, {}px) => ({:.0f}px, {:.0f}px) => ({}, {}) pulse in {:.0f} ms".format(center_to_x, center_to_y, x_px, y_px, x_pulse, y_pulse, t_sec*1000))

        # Case 2
        # # SPEED = 12000 # full speed: 120000, half speed: 600000
        # # SPEEDS = list(map(lambda idx: int(30000 * self.FOVS[idx-1][0]/self.FOVS[0][0]), list(range(0,21))))
        # # [30000, 30000, 15000, 0, 7500, 0, 0, 0, 3750, 0, 0, 0, 2500, 0, 0, 0, 1875, 0, 0, 0, 1500]
        # SPEEDS = [0, 30000, 30000, 0, 30000, 0, 0, 0, 12000, 0, 0, 0, 0, 0, 6000, 0, 6000, 0, 0, 0, 0]
        # SPEED = SPEEDS[current_zoom]
        # # SPEED = 30000
        #
        # (x_pulse, y_pulse, z_pulse, f_pulse) = self.pixel_to_pulse(center_to_x, center_to_y, current_zoom, limit = True)
        # d_pulse = max(abs(x_pulse), abs(y_pulse))
        # t_sec = d_pulse / SPEED # 소요 시간
        # # print("[MOTOR] ({}px, {}px) => ({}, {}) pulse in {:.0f} ms @{:04.0f}".format(center_to_x, center_to_y, x_pulse, y_pulse, t_sec*1000, lap))
        #
        # MIN_DISTANCES = [0, 10, 20, 0, 30, 0, 0, 0, 60, 0, 0, 0, 0, 0, 60, 0, 60, 0, 0, 0, 0]
        # MIN_DISTANCE = MIN_DISTANCES[current_zoom]
        #
        # MIN_MOVING_DISTANCES = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]
        # MIN_MOVING_DISTANCE = MIN_MOVING_DISTANCES[current_zoom]
        #
        # MOVING_TIME = 0.1 # 0.1 for 100 ms
        #
        # if t_sec >= MOVING_TIME:
        #     x_pulse = int(x_pulse * (MOVING_TIME / t_sec))
        #     y_pulse = int(y_pulse * (MOVING_TIME / t_sec))
        #     t_sec = MOVING_TIME
        # elif t_sec >= MOVING_TIME/2:
        #     x_pulse = int(x_pulse * (MOVING_TIME/2 / t_sec))
        #     y_pulse = int(y_pulse * (MOVING_TIME/2 / t_sec))
        #     t_sec = MOVING_TIME
        # else:
        #     t_sec = 0
        #     x_pulse = 0
        #     y_pulse = 0
        #
        # x_px, y_px = self.pulse_to_pixel(x_pulse, y_pulse, current_zoom)
        # distance = math.sqrt(center_to_x**2 + center_to_y**2)
        # # print("[MOTOR] Distance from Center: ({}px, {}px) => {:.0f}".format(center_to_x, center_to_y, distance))
        #
        # if t_sec > 0 and abs(distance - self.prev_distance) > MIN_MOVING_DISTANCE: # and distance > MIN_DISTANCE:
        #     # print("[MOTOR: DO] ({}px, {}px) => ({:.0f}px, {:.0f}px) => ({}, {}) pulse in {:.0f} ms".format(center_to_x, center_to_y, x_px, y_px, x_pulse, y_pulse, t_sec*1000))
        #     self.prev_distance = distance
        #     self.move(x = x_pulse, y = y_pulse, z = z_pulse, f = f_pulse, t = t_sec)
        #     self.is_moving = True
        #     motor_timer = Timer(t_sec*1.2, self.has_finished_moving, args = [False])
        #     motor_timer.start()
        # # else:
        # #     print("[MOTOR: DON'T] ({}px, {}px) => ({:.0f}px, {:.0f}px) => ({}, {}) pulse in {:.0f} ms".format(center_to_x, center_to_y, x_px, y_px, x_pulse, y_pulse, t_sec*1000))

        # Case 3
        # SPEEDS = [0, 30000, 30000, 0, 30000, 0, 0, 0, 15000, 0, 0, 0, 0, 0, 15000, 0, 15000, 0, 0, 0, 0]
        # SPEED = SPEEDS[current_zoom]
        # # SPEED = 30000
        #
        # MIN_DISTANCES = [0, 10, 20, 0, 20, 0, 0, 0, 60, 0, 0, 0, 0, 0, 60, 0, 60, 0, 0, 0, 0]
        # MIN_DISTANCE = MIN_DISTANCES[current_zoom]
        #
        # MIN_MOVING_DISTANCES = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]
        # MIN_MOVING_DISTANCE = MIN_MOVING_DISTANCES[current_zoom]
        #
        # (x_pulse, y_pulse, z_pulse, f_pulse) = self.pixel_to_pulse(center_to_x, center_to_y, current_zoom, limit = True)
        # d_pulse = max(abs(x_pulse), abs(y_pulse))
        # t_sec = d_pulse / SPEED # 소요 시간
        #
        # distance = math.sqrt(center_to_x**2 + center_to_y**2)
        #
        # # print("[MOTOR] ({}px, {}px) => ({}, {}) pulse in {:.0f} ms @{:04.0f}".format(center_to_x, center_to_y, x_pulse, y_pulse, t_sec*1000, lap))
        # if t_sec > 0.05 and abs(distance - self.prev_distance) > MIN_MOVING_DISTANCE and distance > MIN_DISTANCE:
        #     self.prev_distance = distance
        #     self.move(x = x_pulse, y = y_pulse, z = z_pulse, f = f_pulse, t = t_sec)
        #     self.is_moving = True
        #     motor_timer = Timer(t_sec*1.2, self.has_finished_moving, args = [False])
        #     motor_timer.start()

        # Case 4
        # # SPEEDS = [0, 30000, 30000, 0, 30000, 0, 0, 0, 15000, 0, 0, 0, 0, 0, 15000, 0, 15000, 0, 0, 0, 0]
        # # SPEED = SPEEDS[current_zoom]
        # SPEED = 30000
        # MOVING_TIME = 0.05
        # (x_pulse, y_pulse, z_pulse, f_pulse) = self.pixel_to_pulse(center_to_x, center_to_y, current_zoom, limit = True)
        # d_pulse = max(abs(x_pulse), abs(y_pulse))
        # t_sec = d_pulse / SPEED # 소요 시간
        # # print("[MOTOR] ({}px, {}px) => ({}, {}) pulse in {:.0f} ms @{:04.0f}".format(center_to_x, center_to_y, x_pulse, y_pulse, t_sec*1000, lap))
        #
        #
        # if t_sec > MOVING_TIME * 3:
        #     x_pulse = int(x_pulse * (MOVING_TIME / t_sec))
        #     y_pulse = int(y_pulse * (MOVING_TIME / t_sec))
        #     t_sec = MOVING_TIME
        # else:
        #     # if x_pulse > do_not_move_conditions[current_zoom] or x_pulse < -do_not_move_conditions[current_zoom]:
        #     #     x_pulse = int(x_pulse / 10)
        #     # else:
        #     x_pulse = 0
        #
        #     # if y_pulse > do_not_move_conditions[current_zoom] or y_pulse < -do_not_move_conditions[current_zoom]:
        #     #     y_pulse = int(y_pulse / 10)
        #     # else:
        #     y_pulse = 0
        #
        # if y_pulse != 0 or x_pulse != 0:
        #     self.move(x = x_pulse, y = y_pulse, z = z_pulse, f = f_pulse, t = t_sec)
        #     self.is_moving = True
        #     motor_timer = Timer(t_sec*1.2, self.has_finished_moving, args = [False])
        #     motor_timer.start()

    def has_finished_zooming(self, zoom):
        # zoom.stop_zooming()
        self.is_zooming = False
        self.current_zoom = zoom
        print("[ZOOM] End of Zooming") # hl1sqi

    def zoom_x1(self):
        # print('[ZOOM] to x1')
        buffer = [0xff,0x01,0x00,0x40,0x00,0x00,0x41]
        bstr = bytes(buffer)
        self.port.write(bstr)

    def zoom_x20(self):
        # print('[ZOOM] to x20')
        buffer = [0xff,0x01,0x00,0x20,0x00,0x00,0x21]
        bstr = bytes(buffer)
        self.port.write(bstr)

    def zoom_to(self, x, dur=0.1):
        # print('[ZOOM] to', x)
        preset = self.zoom_to_preset[x]
        print('[Debug] x = ', x)

        buffer = [0xff,0x01,0x00,0x07,0x00,preset,0x00]
        checksum = 0
        for el in buffer[1: -1]:
            checksum += el

        checksum = checksum % 256
        buffer[-1] = checksum
        bstr = bytes(buffer)
        self.port.write(bstr)

        self.is_zooming = True
        zoom_timer = Timer(dur, self.has_finished_zooming, args = [x])
        zoom_timer.start()

    def stop_zooming(self):
        buffer = [0xff,0x01,0x00,0x00,0x00,0x00,0x01]
        bstr = bytes(buffer)
        self.port.write(bstr)
        time.sleep(0.1)

    def set_preset(self, num):
        buffer = [0xff,0x01,0x00,0x03,0x00,num,0x00]
        checksum = 0
        for el in buffer[1: -1]:
            checksum += el

        checksum = checksum % 256
        buffer[-1] = checksum
        bstr = bytes(buffer)
        self.port.write(bstr)

    def get_preset(self, num):
        buffer = [0xff,0x01,0x00,0x07,0x00,num,0x00]
        checksum = 0
        for el in buffer[1: -1]:
            checksum += el

        checksum = checksum % 256
        buffer[-1] = checksum
        bstr = bytes(buffer)
        self.port.write(bstr)

    def zoom(self, direction):
        if direction == 'in':
            buffer = [0xff,0x01,0x00,0x20,0x00,0x00,0x21]
            bstr = bytes(buffer)
            self.port.write(bstr)
        else:
            buffer = [0xff,0x01,0x00,0x40,0x00,0x00,0x41]
            bstr = bytes(buffer)
            self.port.write(bstr)
        time.sleep(0.1)
        self.stop_zooming()

    def find_next_zoom(self, dir):
        idx = self.available_zooms.index(self.current_zoom)
        next_zoom = self.current_zoom

        if dir == 'in':
            if idx < len(self.available_zooms) - 1:
                idx += 1
                next_zoom = self.available_zooms[idx]
        elif dir == 'out':
            if idx > 0:
                idx -= 1
                next_zoom = self.available_zooms[idx]
        elif dir == 'first':
            next_zoom = self.available_zooms[0]
        elif dir == 'last':
            next_zoom = self.available_zooms[len(self.available_zooms) - 1]

        return next_zoom

    def find_next_auto_zoom(self, current_length):
        idx = self.available_zooms.index(self.current_zoom)
        zoom_in_idx = idx + 1 if idx < len(self.available_zooms) - 1 else len(self.available_zooms) - 1
        zoom_out_idx = idx - 1 if idx > 0 else 0

        normalized_length = list(map(lambda idx: self.FOVS[0][0]/self.FOVS[idx-1][0], self.available_zooms))
        # print("[ZOOM] Ratio in length", list(map(lambda i: round(i, 2), normalized_length)))

        zoom_in_length = current_length * (normalized_length[zoom_in_idx]/normalized_length[idx])
        zoom_out_length = current_length * (normalized_length[zoom_out_idx]/normalized_length[idx])
        max_length = self.WIDTH * self.scale
        # print("[ZOOM] Current: {0:.0f}/{3}, Zoom in: {1:.0f}/{3}, Zoom out: {2:.0f}/{3}".format(current_length, zoom_in_length, zoom_out_length, self.WIDTH))

        if current_length >= max_length * 1.4:
            next_zoom = self.available_zooms[zoom_out_idx]
        elif current_length > 0 and zoom_in_length < max_length * 1.2: # 줌인할 길이가 상한을 넘지 않은 경우에는 줌인
            next_zoom = self.available_zooms[zoom_in_idx]
        else:
            next_zoom = self.current_zoom

        return next_zoom
