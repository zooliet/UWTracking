
import cv2
import numpy as np
import imutils
from utils import util
import itertools

class KPMTracker:
    def __init__(self):
        print("[KPM] __init__")
        self.detector = cv2.BRISK_create(10)
        # self.detector = cv2.AKAZE_create()
        # self.detector = cv2.xfeatures2d.SIFT_create()
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    def init(self, frame):
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # self.gray0 = np.copy(gray)
        # self.x01 = self.x1
        # self.y01 = self.y1
        # self.x02 = self.x2
        # self.y02 = self.y2
        # mask = np.zeros(gray.shape[:2], dtype=np.uint8)
        # cv2.rectangle(mask, (self.x1, self.y1), (self.x2, self.y2), 255, -1)
        # mask = cv2.bitwise_and(gray, gray, mask=mask)
        # cv2.imshow('Mask', mask)
        #
        # self.kp1, self.desc1 = self.detector.detectAndCompute(gray, mask)
        # # print(len(self.kp1))
        self.force_init_flag = False

    def update(self, frame):
        print("[KPM] update")

    def filter_matches(self, kp1, kp2, matches, ratio = 0.75):
        mkp1, mkp2 = [], []
        for m in matches:
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                m = m[0]
                mkp1.append( kp1[m.queryIdx] )
                mkp2.append( kp2[m.trainIdx] )
        p1 = np.float32([kp.pt for kp in mkp1])
        p2 = np.float32([kp.pt for kp in mkp2])
        kp_pairs = zip(mkp1, mkp2)
        return p1, p2, list(kp_pairs)

    def match(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.kp2, self.desc2 = self.detector.detectAndCompute(gray, None)
        print(len(self.kp2))

        raw_matches = self.matcher.knnMatch(self.desc1, trainDescriptors = self.desc2, k = 2)
        p1, p2, kp_pairs = self.filter_matches(self.kp1, self.kp2, raw_matches)

        if len(p1) >= 4:
            H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
            print('%d / %d  inliers/matched' % (np.sum(status), len(status)))
        else:
            H, status = None, None
            print('%d matches found, not enough for homography estimation' % len(p1))

        # vis = explore_match(win, img1, img2, kp_pairs, status, H)
        # def explore_match(win, img1, img2, kp_pairs, status = None, H = None):
        h1, w1 = self.gray0.shape[:2]
        h2, w2 = gray.shape[:2]

        vis = np.zeros((max(h1, h2), w1+w2), np.uint8)
        vis[:h1, :w1] = self.gray0
        vis[:h2, w1:w1+w2] = gray
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
        if H is not None:
            corners = np.float32([[self.x01, self.y01], [self.x02, self.y01], [self.x02, self.y02], [self.x01, self.y02]])
            corners = np.int32( cv2.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2) + (w1, 0) )
            cv2.polylines(vis, [corners], True, (255, 255, 255))

        if status is None:
            status = np.ones(len(kp_pairs), np.bool_)
        p1, p2 = [], []  # python 2 / python 3 change of zip unpacking
        for kpp in kp_pairs:
            p1.append(np.int32(kpp[0].pt))
            p2.append(np.int32(kpp[1].pt))

        green = (0, 255, 0)
        red = (0, 0, 255)
        white = (255, 255, 255)
        kp_color = (51, 103, 236)

        for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
            if inlier:
                col = green
                cv2.circle(vis, (x1, y1), 2, col, -1)
                cv2.circle(vis, (x2, y2), 2, col, -1)
            else:
                col = red
                r = 2
                thickness = 3
                cv2.line(vis, (x1-r, y1-r), (x1+r, y1+r), col, thickness)
                cv2.line(vis, (x1-r, y1+r), (x1+r, y1-r), col, thickness)
                cv2.line(vis, (x2-r, y2-r), (x2+r, y2+r), col, thickness)
                cv2.line(vis, (x2-r, y2+r), (x2+r, y2-r), col, thickness)

        # vis0 = vis.copy()
        for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
            if inlier:
                cv2.line(vis, (x1, y1), (x2, y2), green)

        cv2.imshow('Keypoints', vis)
