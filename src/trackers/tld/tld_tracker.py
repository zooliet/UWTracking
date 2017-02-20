
import cv2
import numpy as np
import imutils
from utils import util


class TLDTracker:
    # def __init__(self):
    #     self._tracker = cv2.Tracker_create("TLD")

    def init(self, frame):
        self.force_init_flag = False
        self.has_result = True

        # x1 = self.x1
        # x2 = self.x2
        # y1 = self.y1
        # y2 = self.y2
        #
        # mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        # cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        # (x1, y1), (x2, y2) = util.selection_enlarged(mask, x1, y1, x2, y2, ratio=3)
        # cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        # frame = cv2.bitwise_and(frame, frame, mask=mask)
        # cv2.imshow('Mask', frame)
        self._tracker = cv2.Tracker_create("TLD")

        bbox = (self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1)
        ok = self._tracker.init(frame, bbox)
        return ok

    def update(self, frame):
        # x1 = self.x1
        # x2 = self.x2
        # y1 = self.y1
        # y2 = self.y2

        # if self.has_result:
        #     mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        #     cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        #     (x1, y1), (x2, y2) = util.selection_enlarged(mask, x1, y1, x2, y2, ratio=3)
        #     cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        #     frame = cv2.bitwise_and(frame, frame, mask=mask)
        #     cv2.imshow('Mask', frame)

        ok, bbox = self._tracker.update(frame)
        if ok:
            self.has_result = True
            self.tl = (int(bbox[0]), int(bbox[1]))
            self.br = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        else:
            self.has_result = False
