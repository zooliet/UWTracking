
import cv2
import numpy as np
from utils import util

class ColorTracker():

    CENTER_DISPLACEMENT = 50
    LOST_CONDITION = 100 # 30ms * 100 = 3 sec
    FOUND_CONDITION = 1

    def __init__(self):
        # red
        self.lower = np.array([0, 150, 100], dtype = "uint8")
        self.upper = np.array([5, 255, 255], dtype = "uint8")
        # self.lower = np.array([165, 90, 90], dtype = "uint8")
        # self.upper = np.array([180, 255, 255], dtype = "uint8")
        # yellow
        # self.lower = np.array([15, 120, 120], dtype = "uint8")
        # self.upper = np.array([35, 255, 255], dtype = "uint8")
    def init(self, frame, options):
        x1 = options['x1']
        x2 = options['x2']
        y1 = options['y1']
        y2 = options['y2']
        self.consecutive_lost = 0

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # hsv[:,:,2] = cv2.equalizeHist(hsv[:,:,2])

        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        (x1, y1), (x2, y2) = util.selection_enlarged(mask, x1, y1, x2, y2, ratio=1.5)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        hsv = cv2.bitwise_and(hsv, hsv, mask=mask)

        binary = cv2.inRange(hsv, self.lower, self.upper)
        binary = cv2.GaussianBlur(binary, (11, 11), 0)
        # cv2.imshow('Binary', binary)

        (_, contours, _) = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]

            (x, y, w, h) = cv2.boundingRect(contour)
            # cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)

            self.center = np.int32([x + w//2, y + h//2])
            # rect = np.int32(cv2.boxPoints(cv2.minAreaRect(contour)))
            # cv2.drawContours(frame, [rect], -1, (255, 0, 0), 2)
            # center = np.mean(rect, axis=0).astype(np.uint32)
            # cv2.drawMarker(frame, tuple(center), (0,255,255))

            return True
        else:
            cx = options['x1'] + (options['x2'] - options['x1']) // 2
            cy = options['y1'] + (options['y2'] - options['y1']) // 4 # approx
            self.center = np.array([cx, cy], dtype=np.uint32)
            return False


    def update(self, frame, options = {'x1': 0, 'y1':0, 'x2': 640, 'y2': 360}):
        x1 = options['x1']
        x2 = options['x2']
        y1 = options['y1']
        y2 = options['y2']

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # hsv[:,:,2] = cv2.equalizeHist(hsv[:,:,2])

        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        (x1, y1), (x2, y2) = util.selection_enlarged(mask, x1, y1, x2, y2, ratio=1.5)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        hsv = cv2.bitwise_and(hsv, hsv, mask=mask)

        binary = cv2.inRange(hsv, self.lower, self.upper)
        binary = cv2.GaussianBlur(binary, (11, 11), 0)

        (_, contours, _) = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # print("[COLOR] I found {} contours".format(len(contours)))
        cv2.drawContours(frame, contours, -1, (255, 0, 0), 1)

        num_of_contours = len(contours)
        if num_of_contours > 0:
            sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
            contour0 = sorted_contours[0]

            (x0, y0, w0, h0) = cv2.boundingRect(contour0)
            center0 = np.int32([x0 + w0//2, y0 + h0//2])
            dist0 = util.distance(center0, self.center)
            area0 = int(w0 * h0)

            if num_of_contours > 1:
                contour1 = sorted_contours[1]
                (x1, y1, w1, h1) = cv2.boundingRect(contour1)
                center1 = np.int32([x1 + w1//2, y1 + h1//2])
                dist1 = util.distance(center1, self.center)
                area1 = int(w1 * h1)
            else:
                dist1 = dist0
                center1 = center0
                area1 = area0
                (x1, y1, w1, h1) = (x0, y0, w0, h0)

            # print("[COLOR] prev center({}) to largest({}) and to the next largest({})".format(self.cener, dist0, dist1))
            if dist0 <= dist1 and dist0 < self.CENTER_DISPLACEMENT:
                self.center = center0
                cv2.rectangle(frame, (x0,y0), (x0+w0, y0+h0), (255,0,0), 2)
                self.area = area0
                self.consecutive_lost = 0
            elif dist0 > dist1 and dist1 < self.CENTER_DISPLACEMENT:
                self.center = center1
                cv2.rectangle(frame, (x1,y1), (x1+w1, y1+h1), (255,0,0), 2)
                self.area = area1
                self.consecutive_lost = 0
            elif self.consecutive_lost > self.LOST_CONDITION: # 100
                if dist0 <= dist1:
                    self.center = center0
                    self.area = area0
                    cv2.rectangle(frame, (x0,y0), (x0+w0, y0+h0), (255,0,0), 2)
                else:
                    self.center = center1
                    self.area = area1
                    cv2.rectangle(frame, (x1,y1), (x1+w1, y1+h1), (255,0,0), 2)
                self.consecutive_lost = 0
            else: # dist0 or dist1 > self.CENTER_DISPLACEMEN
                self.consecutive_lost += 1

        else:
            self.consecutive_lost += 1
