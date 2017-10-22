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

class Zoom:
    def __init__(self, dev, baud = 115200, screen_width = 640):
        if dev:
            self.port = serial.Serial(dev, baud, timeout = 0, parity = serial.PARITY_NONE)
        else:
            self.port = None

        self.WIDTH = screen_width
        self.HEIGHT = int(screen_width * 9 / 16)

        self.FOCUSING_DURATION = 1.0 # 2.3 # sec

        self.VISCA_ZOOMS = {
            1: 0x0000,  1.4: 0x0ccd, 1.7: 0x1333,
            2: 0x187c,  3: 0x22fa,  4: 0x28ec,  5: 0x2d76,
            6: 0x309c,  7: 0x3310,  8: 0x3582,  9: 0x3742,  10: 0x38a8,
            11: 0x3a0e, 12: 0x3b74, 13: 0x3c80, 14: 0x3d34, 15: 0x3de6,
            16: 0x3e9a, 17: 0x3f26, 18: 0x3f8c, 19: 0x3fcc, 20: 0x4000
        }

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

        self.auto_scale = 0.4

        self.is_zooming = False
        self.current_zoom = 1
        self.zoom_to(1)

    def has_finished_zooming(self, zoom):
        self.current_zoom = zoom
        self.is_zooming = False
        # current_time = datetime.datetime.now().time().isoformat()
        # print("[ZOOM] End of Zooming: {}".format(current_time)) # hl1sqi

    def recv_packet(self):
        pass
        # packet=''
        # count=0
        # while count<16:
        #     s = self.port.read(1)
        #     if s:
        #         byte = ord(s)
        #         count+=1
        #         packet = packet + chr(byte)
        #     else:
        #         print("ERROR: Timeout waiting for reply")
        #         break
        #     if byte==0xff:
        #         break
        # return packet

    def zoom_to(self, zoom, dur=0.1):
        if self.port and self.is_zooming is False:
            ms = (self.VISCA_ZOOMS[zoom] &  0b1111111100000000) >> 8
            ls = (self.VISCA_ZOOMS[zoom] &  0b0000000011111111)
            p= (ms & 0b11110000) >> 4
            r= (ls & 0b11110000) >> 4
            q = ms & 0b1111
            s= ls & 0b1111
            # print(p, r, q, s)
            buffer = [0x81, 0x01, 0x04, 0x47, p, q, r, s, 0xFF]
            # print(buffer)
            bstr = bytes(buffer)
            # print(bstr)
            self.port.write(bstr)
            self.port.write(bstr)
            # reply = self.recv_packet()
            # if reply[-1:] == '\xff':
            #     self.current_zoom = zoom

            zoom_timer = Timer(dur, self.has_finished_zooming, args = [zoom])
            zoom_timer.start()
            self.is_zooming = True
            # self.current_zoom = zoom

    def zoom_in(self):
        if self.current_zoom == 16:
            pass
        elif self.current_zoom == 1:
            self.zoom_to(1.4)
        elif self.current_zoom == 1.4:
            self.zoom_to(1.7)
        elif self.current_zoom == 1.7:
            self.zoom_to(2)
        else:
            self.zoom_to(self.current_zoom + 1)

    def zoom_out(self):
        if self.current_zoom == 1:
            pass
        elif self.current_zoom == 2:
            self.zoom_to(1.7)
        elif self.current_zoom == 1.7:
            self.zoom_to(1.4)
        elif self.current_zoom == 1.4:
            self.zoom_to(1)
        else:
            self.zoom_to(self.current_zoom - 1)

    def autozoom(self, width, height):
        # print('Auto: {}, {}'.format(width, height))
        if self.port and self.is_zooming is False:
            next_zoom = self.find_next_zoom(width, height)
            if self.current_zoom != next_zoom:
                self.zoom_to(next_zoom, dur=self.FOCUSING_DURATION)

    def find_next_zoom(self, width, height):
        zooms = [1, 1.4, 1.7, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

        idx = zooms.index(self.current_zoom)
        zoom_in_idx = idx + 1 if idx < len(zooms) - 1 else len(zooms) - 1
        zoom_out_idx = idx - 1 if idx > 0 else 0

        normalized_length = list(map(lambda zoom: self.FOVS[1][0]/self.FOVS[zoom][0], zooms))
        normalized_height = list(map(lambda zoom: self.FOVS[1][1]/self.FOVS[zoom][1], zooms))
        # print("[ZOOM] Ratio in length", list(map(lambda l: round(l, 2), normalized_length)))

        zoom_in_length = width * (normalized_length[zoom_in_idx]/normalized_length[idx])
        zoom_out_length = width * (normalized_length[zoom_out_idx]/normalized_length[idx])
        max_length = self.WIDTH * self.auto_scale

        zoom_in_height = height * (normalized_height[zoom_in_idx]/normalized_height[idx])

        # print("[ZOOM] Current: {0:.0f}/{3}, Zoom in: {1:.0f}/{3}, Zoom out: {2:.0f}/{3}".format(width, zoom_in_length, zoom_out_length, self.WIDTH))

        next_zoom = self.current_zoom
        if width >= max_length * 1.4:
            next_zoom = zooms[zoom_out_idx]
        elif width > 0 and zoom_in_length < max_length * 1.2: # 줌인할 길이가 상한을 넘지 않은 경우
            if zoom_in_height < self.HEIGHT * 0.9: # 줌인할 높이가 높이 상한을 넘지 않을 경우
                next_zoom = zooms[zoom_in_idx]

        return next_zoom

    # For testing only
    def zoom(self, value):
        if self.port and self.is_zooming is False:
            ms = (value &  0b1111111100000000) >> 8
            ls = (value &  0b0000000011111111)
            p= (ms & 0b11110000) >> 4
            r= (ls & 0b11110000) >> 4
            q = ms & 0b1111
            s= ls & 0b1111
            # print(p, r, q, s)
            buffer = [0x81, 0x01, 0x04, 0x47, p, q, r, s, 0xFF]
            # print(buffer)
            bstr = bytes(buffer)
            # print(bstr)
            self.port.write(bstr)
            # self.current_zoom = zoom

    # For testing only
    def focus(self, value):
        if self.port and self.is_zooming is False:
            ms = (value &  0b1111111100000000) >> 8
            ls = (value &  0b0000000011111111)
            t= (ms & 0b11110000) >> 4
            u= (ls & 0b11110000) >> 4
            v = ms & 0b1111
            w= ls & 0b1111
            # print(p, r, q, s)
            buffer = [0x81, 0x01, 0x04, 0x48, t, u, v, w, 0xFF]
            # print(buffer)
            bstr = bytes(buffer)
            # print(bstr)
            self.port.write(bstr)
            # self.current_zoom = zoom


    # Pelco-d only
    def zoom_x1(self):
        if self.port and self.is_zooming is False:
            # print('[ZOOM] to x1')
            buffer = [0xff,0x01,0x00,0x40,0x00,0x00,0x41]
            bstr = bytes(buffer)
            self.port.write(bstr)
            self.current_zoom = 1

    # Pelco-d only
    def zoom_x20(self):
        if self.port and self.is_zooming is False:
            # print('[ZOOM] to x20')
            buffer = [0xff,0x01,0x00,0x20,0x00,0x00,0x21]
            bstr = bytes(buffer)
            self.port.write(bstr)
            self.current_zoom = 20
