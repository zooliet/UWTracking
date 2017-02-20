
import cv2
import numpy as np
import imutils
from utils import util
import itertools

class MotionTracker:
    def __init__(self):
        self.interval = 100
        self.motion_count = 0

    def init(self, count):
        self.motion_count = count

    def update(self, frame, prev_frame):
        self.motion_count = 0

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7,7), 0)
        prev_gray = cv2.GaussianBlur(prev_gray, (7,7), 0)

        diffed = cv2.absdiff(prev_gray, gray)

        # 아래의 파라메터는 설계 인자
        T1 = 10
        T2 = 50
        BLUR_SIZE = (3, 3)

        (T, thresholded) = cv2.threshold(diffed, T1, 255, cv2.THRESH_BINARY)
        blurred = cv2.GaussianBlur(thresholded, BLUR_SIZE, 0)
        (T, rethresholded) = cv2.threshold(blurred, T2, 255, cv2.THRESH_BINARY)
        (_, cnts, _) = cv2.findContours(rethresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # mean_thresholded = cv2.adaptiveThreshold(diffed, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 4)
        # blurred = cv2.GaussianBlur(mean_thresholded, BLUR_SIZE, 0)
        # (_, cnts, _) = cv2.findContours(mean_thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.imshow('Compare', np.hstack((thresholded, rethresholded, mean_thresholded)))

        # print("I count {} contours in this image".format(len(cnts)))
        cv2.drawContours(frame, cnts, -1, (0, 0, 255), 10)
        # for (i, c) in enumerate(cnts):
        #     if cv2.contourArea(c) > 20:
        #         cv2.drawContours(frame, [c], -1, (0, 0, 255), 10)

        binary = cv2.inRange(frame, (0,0,255), (0,0,255))
        binary = cv2.GaussianBlur(binary, (11, 11), 0)
        (_, contours, _) = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]
            (x, y, w, h) = cv2.boundingRect(contour)
            # cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
            # rect = np.int32(cv2.boxPoints(cv2.minAreaRect(contour)))
            # cv2.drawContours(frame, [rect], -1, (0, 255, 0), 2)
            # cv2.imshow('Motion', frame)
            # return x, y, x+w, y+h

            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            (x1, y1), (x2, y2) = util.selection_enlarged(mask, x, y, x+w, y+h, ratio=0.8)
            # print(x, y, x+w, y+h, '=>', x1, y1, x2, y2)
            return x1, y1, x2, y2
        else:
            return -1, -1, -1, -1

        # centers = []
        # for (i, c) in enumerate(cnts):
        #     (x, y, w, h) = cv2.boundingRect(c)
        #     cX = x + w//2
        #     cY = y + h//2
        #     # cv2.drawMarker(frame, (cX, cY), (0,255,0))
        #
        #     M = cv2.moments(c)
        #     if int(M["m00"]) >= 2: #and int(M["m00"]) < 100: # cv2.contourArea(c) > 2:
        #         cX = int(M["m10"] / M["m00"])
        #         cY = int(M["m01"] / M["m00"])
        #         centers.append((cX, cY))
        #         # D = dist.euclidean((self.cX, self.cY), (cX, cY))
        #         # if D < 50:
        #         #     centers.append((cX, cY))
        #     else:
        #         print(M["m00"]) # Too small
        #
        # if len(centers) > 0:
        #     res = np.array(centers)
        #     self.cX = int(res[:,0].mean())
        #     self.cY = int(res[:,1].mean())
        #     self.x1 = self.cX - (self.w // 2)
        #     self.y1 = self.cY - (self.h // 2)

    def check_interval(self):
        self.motion_count += 1
        return self.motion_count % self.interval == 0
