import struct
import serial
import time
from threading import Timer

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
        self.available_zooms = [1,2,4,8,16]
        self.current_zoom = 1

        self.WIDTH = screen_width # 640x360, 1024x576, 1280x720, 1920x1080
        self.HEIGHT = int(self.WIDTH * 9 / 16)
        self.HALF_WIDTH = self.WIDTH // 2
        self.HALF_HEIGHT = self.HEIGHT // 2

        self.DEGREE_PER_PULSE = 0.00048 # 0.00048은 현재 사용 모터와 기어비로 결정되는 펄스 당 회전 각도 (degree)

        # FOVS 는 실제로는 Half FOVS를 표현
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
                    (0, 0),
                    (0, 0),
                    (62.5000/32, 34.5000/32), #16
                    (0, 0),
                    (0, 0),
                    (0, 0),
                    (62.5000/40, 34.5000/40), ] #20

    def has_finished_moving(self, args):
        self.is_moving = False
        # print("[MOTOR] End of Moving")

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

        if (self.sum_of_x_degree > 90 and x > 0) or (self.sum_of_x_degree < -90 and x < 0):
            x = 0
        else:
            self.sum_of_x_degree += (x * self.DEGREE_PER_PULSE)

        if (self.sum_of_y_degree > 90 and y > 0) or (self.sum_of_y_degree < -90 and y < 0):
            y = 0
        else:
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

    def pixel_to_pulse(self, x_px, y_px, zoom = 1, limit = False):
        # Logitec
        # x_degree = x_px/320 * 66.59215896/2
        # y_degree = y_px/180 * 40.55232596/2

        # UWTec
        # x_degree = x_px/self.HALF_WIDTH * 62.0000/2
        # y_degree = y_px/self.HALF_HEIGHT * 34.5000/2

        x_degree = x_px/self.HALF_WIDTH * self.FOVS[zoom-1][0]
        y_degree = y_px/self.HALF_HEIGHT * self.FOVS[zoom-1][1]

        x = int(x_degree/self.DEGREE_PER_PULSE)
        y = int(y_degree/self.DEGREE_PER_PULSE)
        z = f = 0

        # print("[MOTOR] ({}px, {}px) => ({:.4f}°, {:.4f}°) => ({}, {}) pulse".format(x_px, y_px, x_degree, y_degree, x, y))
        return x, y, z, f

    def move_to(self, x, y, current_zoom=1):
        motor_timer = Timer(1, self.has_finished_moving, args = [False])
        (x_to, y_to, z_to, f_to) = self.pixel_to_pulse(x, y, current_zoom, limit = False)
        self.move(x = x_to, y = y_to, z = z_to, f = f_to, t = 1)
        motor_timer.start()
        self.is_moving = True

    def track(self, center_to_x, center_to_y, current_zoom=1):
        if abs(center_to_x) > 2 or abs(center_to_y) > 2:
            (x_to, y_to, z_to, f_to) = self.pixel_to_pulse(center_to_x, center_to_y, current_zoom, limit = True)

            # SPEED = 12000 # full speed: 120000, half speed: 600000
            # SPEEDS = list(map(lambda idx: int(120000 * self.FOVS[idx-1][0]/self.FOVS[0][0]), list(range(0,21))))
            # SPEEDS = [3000, 60000, 30000, 0, 15000, 0, 0, 0, 7500, 0, 0, 0, 5000, 0, 0, 0, 3750, 0, 0, 0, 3000]
            SPEEDS = [6000, 60000, 30000, 0, 30000, 0, 0, 0, 15000, 0, 0, 0, 10000, 0, 0, 0, 7500, 0, 0, 0, 6000]
            # print(SPEEDS)
            SPEED = SPEEDS[current_zoom]
            MAX_MOVING_TIME = 0.05 # 0.1 for 100 ms
            d = max(abs(x_to), abs(y_to))
            t_sec = d / SPEED

            do_not_move_conditions = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

            if t_sec > MAX_MOVING_TIME * 3:
                x_to = int(x_to * (MAX_MOVING_TIME / t_sec))
                y_to = int(y_to * (MAX_MOVING_TIME / t_sec))
                t_sec = MAX_MOVING_TIME
            else:
                if x_to > do_not_move_conditions[current_zoom] or x_to < -do_not_move_conditions[current_zoom]:
                    x_to = int(x_to / 10)
                else:
                    x_to = 0

                if y_to > do_not_move_conditions[current_zoom] or y_to < -do_not_move_conditions[current_zoom]:
                    y_to = int(y_to / 10)
                else:
                    y_to = 0

            if y_to == 0 and x_to == 0:
                t_sec = 0

            self.move(x = x_to, y = y_to, z = z_to, f = f_to, t = t_sec)

            if t_sec > 0:
                self.is_moving = True
                motor_timer = Timer(t_sec, self.has_finished_moving, args = [False])
                motor_timer.start()

    def has_finished_zooming(self, zoom):
        # zoom.stop_zooming()
        self.is_zooming = False
        self.current_zoom = zoom
        # print("[ZOOM] End of Zooming")

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
        zoom_to_preset = {1: 1, 2: 2, 4: 3, 8: 4, 16: 5} # zoom to preset
        # print('[Debug] x = ', x)
        preset = zoom_to_preset[x]

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

    def find_next_auto_zoom(self, target_length):
        idx = self.available_zooms.index(self.current_zoom)
        zoom_in_idx = idx + 1 if idx < len(self.available_zooms) - 1 else len(self.available_zooms) - 1
        zoom_out_idx = idx - 1 if idx > 0 else 0

        normalized_length = list(map(lambda idx: self.FOVS[0][0]/self.FOVS[idx-1][0], self.available_zooms))
        # print("[ZOOM] Ratio in length", list(map(lambda i: round(i, 2), normalized_length)))

        zoom_in_length = target_length * (normalized_length[zoom_in_idx]/normalized_length[idx])
        zoom_out_length = target_length * (normalized_length[zoom_out_idx]/normalized_length[idx])
        max_length = self.WIDTH * 0.3
        # print("[ZOOM] Current: {:02.0f}, Zoom in: {:02.0f}, Zoom out: {:02.0f}".format(target_length, zoom_in_length, zoom_out_length))

        if target_length >= max_length + 20:
            next_zoom = self.available_zooms[zoom_out_idx]
        elif target_length > 0 and zoom_in_length < max_length: # 줌인할 길이가 상한을 넘지 않은 경우에는 줌인
            next_zoom = self.available_zooms[zoom_in_idx]
        else:
            next_zoom = self.current_zoom

        return next_zoom


# FOVS_CAM_1 = [(62.0000/2, 34.5000/2), #1
#             (28.5800/2, 16.5000/2), #2
#             (20.6666/2, 11.5000/2),
#             (14.4000/2, 8.0000/2), #4
#             (12.4000/2, 6.9000/2),
#             (10.3333/2, 5.7500/2),
#             (8.8571/2, 4.9286/2),
#             (7.5000/2, 4.1000/2), #8
#             (6.8888/2, 3.8333/2),
#             (6.2000/2, 3.4500/2),
#             (5.6364/2, 3.1364/2),
#             (5.1667/2, 2.8750/2), #12
#             (4.7692/2, 2.6538/2),
#             (4.4286/2, 2.4643/2),
#             (4.1333/2, 2.3000/2),
#             (3.8000/2, 2.1562/2), #16
#             (3.6471/2, 2.1000/2),
#             (3.4444/2, 2.0294/2),
#             (3.2632/2, 1.9167/2),
#             (3.2000/2, 1.8000/2), ] #20
#
# # serial: 16092601
# FOVS_16092601 = [(62.5000/2, 34.5000/2), #1
#             (28.0000/2, 15.4560/2), #2
#             (20.6666/2, 11.5000/2),
#             (14.8500/2, 8.1972/2), #4
#             (12.4000/2, 6.9000/2),
#             (10.3333/2, 5.7500/2),
#             (8.8571/2, 4.9286/2),
#             (7.8500/2, 4.3332/2), #8
#             (6.8888/2, 3.8333/2),
#             (6.2000/2, 3.4500/2),
#             (5.6364/2, 3.1364/2),
#             (5.2000/2, 2.8704/2), #12
#             (4.7692/2, 2.6538/2),
#             (4.4286/2, 2.4643/2),
#             (4.1333/2, 2.3000/2),
#             (3.9500/2, 2.1804/2), #16
#             (3.6471/2, 2.1000/2),
#             (3.4444/2, 2.0294/2),
#             (3.2632/2, 1.9167/2),
#             (3.5000/2, 1.9320/2), ] #20
