import cv2
import numpy as np
import imutils
from utils import util

import itertools
import scipy.spatial
import scipy.cluster

lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict( maxCorners = 500,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
class KLTTracker:
    def __init__(self):
        print("[KLT] __init__")
        self.track_len = 100
        self.detect_interval = 10
        self.tracks = []
        self.frame_idx = 0


    def init(self, frame):
        # print("[KLT] init()")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # mask = np.zeros(gray.shape[:2], dtype=np.uint8)
        # cv2.rectangle(mask, (self.x1, self.y1), (self.x2, self.y2), 255, -1)
        # mask = cv2.bitwise_and(gray, gray, mask=mask)

        mask = np.zeros_like(gray)
        mask[:] = 255

        for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
            cv2.circle(mask, (x, y), 5, 0, -1)

        # cv2.imshow('Mask', mask)

        p = cv2.goodFeaturesToTrack(gray, mask = mask, **feature_params)
        if p is not None:
            for x, y in np.float32(p).reshape(-1, 2):
                self.tracks.append([(x, y)])

        self.frame_idx = 1


    def update(self, frame, prev_frame):
        # print("[KLT] update()")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

        p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
        p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **lk_params)
        p0r, st, err = cv2.calcOpticalFlowPyrLK(gray, prev_gray, p1, None, **lk_params)

        d = abs(p0-p0r).reshape(-1, 2).max(-1)
        good = d < 1
        new_tracks = []
        for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
            if not good_flag:
                continue
            tr.append((x, y))
            if len(tr) > self.track_len:
                del tr[0]
            new_tracks.append(tr)
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
        self.tracks = new_tracks

        # p_first = np.int32([tr[0] for tr in self.tracks]).reshape(-1, 1, 2)
        # p_last = np.int32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
        # d = abs(p_first-p_last).reshape(-1, 2).max(-1)
        # good = d > 5
        #
        # for (x,y), good_flag in zip(p_last.reshape(-1, 2), good):
        #     if good_flag:
        #         cv2.circle(frame, (x, y), 10, (255, 255, 255), -1)

        # p_firsts = np.int32([tr[0] for tr in self.tracks])
        # p_lasts = np.int32([tr[-1] for tr in self.tracks])
        # diff = p_lasts - p_firsts
        # d = np.sqrt((diff ** 2).sum(axis=1))
        # good = d > 5
        # p_lasts = p_lasts[good]
        #
        # util.draw_keypoints(p_lasts, frame, (255, 255, 255), 10, -1)

        cv2.polylines(frame, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))
        util.draw_str(frame, (20, 20), 'track count: %d' % len(self.tracks))
        cv2.imshow('Points', frame)

        self.frame_idx += 1
