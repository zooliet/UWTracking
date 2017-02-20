
import cv2
import numpy as np
from utils import util

class ColorTracker():

    CENTER_DISPLACEMENT = 100
    LOST_CONDITION = 50 # 30ms * 100 = 3 sec
    FOUND_CONDITION = 2

    def __init__(self, width = 640, height = 360):
        self.interval = 100
        self.frame_count = 0

        self.WIDTH = width
        self.HEIGHT = height

        # red
        self.lower = np.array([0, 120, 120], dtype = "uint8")
        self.upper = np.array([5, 255, 255], dtype = "uint8")

        # self.lower = np.array([0, 150, 100], dtype = "uint8")
        # self.upper = np.array([5, 255, 255], dtype = "uint8")
        # self.lower = np.array([165, 90, 90], dtype = "uint8")
        # self.upper = np.array([180, 255, 255], dtype = "uint8")
        # yellow
        # self.lower = np.array([15, 120, 120], dtype = "uint8")
        # self.upper = np.array([35, 255, 255], dtype = "uint8")


    def init(self, count):
        self.frame_count = count

    def update(self, frame, options = {'x1': 0, 'y1':0, 'x2': 640, 'y2': 360}, find_by_area=False):
        x1 = options['x1']
        x2 = options['x2']
        y1 = options['y1']
        y2 = options['y2']

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv = cv2.GaussianBlur(hsv, (7, 7), 0)
        # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # hsv[:,:,2] = cv2.equalizeHist(hsv[:,:,2])

        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        # (x1, y1), (x2, y2) = util.selection_enlarged(mask, x1, y1, x2, y2, ratio=1.5)
        hsv = cv2.bitwise_and(hsv, hsv, mask=mask)
        # binary = cv2.inRange(hsv, self.lower, self.upper)

        min_red = np.array([0, 120, 120])
        max_red = np.array([5, 256, 256])
        binary1 = cv2.inRange(hsv, min_red, max_red)

        min_red = np.array([170, 120, 120])
        max_red = np.array([180, 256, 256])
        binary2 = cv2.inRange(hsv, min_red, max_red)

        binary = binary1 + binary2
        # binary = cv2.GaussianBlur(binary, (11, 11), 0)
        # cv2.imshow('Binary', binary)

        (_, contours, _) = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # print("[COLOR] I found {} contours".format(len(contours)))
        # cv2.drawContours(frame, contours, -1, (255, 0, 0), 1)
        # cv2.imshow('Contour', frame)

        self.center = (np.nan, np.nan)

        if len(contours) > 0:
            if find_by_area:
                sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
                contour_largest = sorted_contours[0]

                (x, y, w, h) = cv2.boundingRect(contour_largest)
                self.center = np.int32([x + w//2, y + h//2])
            else:
                contour_distances = [(self.distance_from_center(contour), contour) for contour in contours]
                closest = min(contour_distances, key=lambda x: x[0])
                closest_distance = closest[0]
                # print("[COLOR] distance is ", closest_distance)

                closest_contour = closest[1]
                (x, y, w, h) = cv2.boundingRect(closest_contour)
                self.center = np.int32([x + w//2, y + h//2])

        # num_of_contours = len(contours)
        # if num_of_contours > 0:
        #     sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
        #     contour_largest = sorted_contours[0]
        #
        #     (x0, y0, w0, h0) = cv2.boundingRect(contour_largest)
        #     center0 = np.int32([x0 + w0//2, y0 + h0//2])
        #     dist0 = util.distance(center0, self.center)
        #
        #     if num_of_contours > 1:
        #         contour_second_largest = sorted_contours[1]
        #         (x1, y1, w1, h1) = cv2.boundingRect(contour_second_largest)
        #         center1 = np.int32([x1 + w1//2, y1 + h1//2])
        #         dist1 = util.distance(center1, self.center)
        #     else:
        #         dist1 = dist0
        #         center1 = center0
        #         (x1, y1, w1, h1) = (x0, y0, w0, h0)
        #
        #     # print("[COLOR] prev center({}) to largest({}) and to the next largest({})".format(self.cener, dist0, dist1))
        #     if dist0 <= dist1 and dist0 < self.CENTER_DISPLACEMENT:
        #         self.center = center0
        #         # cv2.rectangle(frame, (x0,y0), (x0+w0, y0+h0), (255,0,0), 2)
        #         self.consecutive_lost = 0
        #         self.consecutive_found += 1
        #         self.x1 = x0
        #         self.y1 = y0
        #         self.x2 = x0+w0
        #         self.y2 = y0+h0
        #     else: # elif dist0 > dist1: # and dist1 < self.CENTER_DISPLACEMENT:
        #         self.center = center1
        #         # cv2.rectangle(frame, (x1,y1), (x1+w1, y1+h1), (255,0,0), 2)
        #         self.consecutive_lost = 0
        #         self.consecutive_found += 1
        #         self.x1 = x1
        #         self.y1 = y1
        #         self.x2 = x1+w1
        #         self.y2 = y1+h1
        #     # elif self.consecutive_lost > self.LOST_CONDITION: # 100
        #     #     if dist0 <= dist1:
        #     #         self.center = center0
        #     #         cv2.rectangle(frame, (x0,y0), (x0+w0, y0+h0), (255,0,0), 2)
        #     #     else:
        #     #         self.center = center1
        #     #         cv2.rectangle(frame, (x1,y1), (x1+w1, y1+h1), (255,0,0), 2)
        #     #     self.consecutive_lost = 0
        #     # else: # dist0 or dist1 > self.CENTER_DISPLACEMEN
        #     #     self.consecutive_lost += 1
        #
        # else:
        #     self.consecutive_lost += 1
        #     self.consecutive_found = 0

    def distance_from_center(self, contour):
        (x, y, w, h) = cv2.boundingRect(contour)
        center = np.int32([x + w//2, y + h//2])
        dist = util.distance(center, np.array([self.WIDTH//2, self.HEIGHT//2]))
        return dist

    def check_interval(self):
        self.frame_count += 1
        if self.frame_count % self.interval == 0:
            self.frame_count = 0
            return True
        else:
            return False
