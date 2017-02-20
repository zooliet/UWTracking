
import cv2
import numpy as np
import imutils
from utils import util
import dlib
import itertools

class DLIBTracker:
    def __init__(self):
        self._tracker = dlib.correlation_tracker()
        self.detector = cv2.BRISK_create(10)
        # self.detector = cv2.AKAZE_create()
        # self.detector = cv2.xfeatures2d.SIFT_create()
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    def init(self, frame):
        self._tracker.start_track(frame, dlib.rectangle(self.x1, self.y1, self.x2, self.y2))
        self.force_init_flag = False
        self.enable = True

    def update(self, frame, options = None):
        if options is None:
            score = self._tracker.update(frame)
        else:
            x1 = options['x1']
            x2 = options['x2']
            y1 = options['y1']
            y2 = options['y2']
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            # cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
            (x1, y1), (x2, y2) = util.selection_enlarged(mask, x1, y1, x2, y2, ratio=1)
            # cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
            # frame = cv2.bitwise_and(frame, frame, mask=mask)
            # cv2.imshow('Mask', frame)
            score = self._tracker.update(frame, dlib.rectangle(x1, y1, x2, y2))

        # print("[DLIB] score:", score)
        rect = self._tracker.get_position()
        return score, int(rect.left()), int(rect.top()), int(rect.right()), int(rect.bottom())
