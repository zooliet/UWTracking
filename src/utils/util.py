# -*- coding: utf-8 -*-

from __future__ import print_function
import sys
PY3 = sys.version_info[0] == 3
if PY3:
    xrange = range

import cv2
import numpy as np

def selection_enlarged(mask, x1, y1, x2, y2, ratio):
    frame_height, frame_width = mask.shape

    w = int((x2 - x1) * ratio)
    h = int((y2 - y1) * ratio)

    cx = (x2 + x1) // 2
    cy = (y2 + y1) // 2

    x1 = max(cx - (w // 2), 0)
    x2 = min(cx + (w // 2), frame_width)
    y1 = max(cy - (h // 2), 0)
    y2 = min(cy + (h // 2), frame_height)

    return (x1, y1), (x2, y2)


def distance(pt1, pt2):
    diff = pt2 - pt1
    dist = np.sqrt((diff**2).sum())
    # print(pt1, pt2, dist)
    return dist


def in_rect(keypoints, tl, br):
    if type(keypoints) is list:
        keypoints = keypoints_cv_to_np(keypoints)

    x = keypoints[:, 0]
    y = keypoints[:, 1]

    C1 = x > tl[0]
    C2 = y > tl[1]
    C3 = x < br[0]
    C4 = y < br[1]

    result = C1 & C2 & C3 & C4

    return result

def out_of_rect(keypoints, tl, br):
    if type(keypoints) is list:
        keypoints = keypoints_cv_to_np(keypoints)

    x = keypoints[:, 0]
    y = keypoints[:, 1]

    C1 = x < tl[0]
    C2 = y < tl[1]
    C3 = x > br[0]
    C4 = y > br[1]

    result = C1 | C2 | C3 | C4

    return result

def keypoints_cv_to_np(keypoints_cv):
    keypoints = np.array([k.pt for k in keypoints_cv])
    return keypoints

def squeeze_pts(X):
    X = X.squeeze()
    if len(X.shape) == 1:
        X = np.array([X])
    return X

def L2norm(X):
    return np.sqrt((X ** 2).sum(axis=1))

def rotate(pt, rad):
    if(rad == 0):
        return pt

    pt_rot = np.empty(pt.shape)

    s, c = [f(rad) for f in (np.math.sin, np.math.cos)]

    pt_rot[:, 0] = c * pt[:, 0] - s * pt[:, 1]
    pt_rot[:, 1] = s * pt[:, 0] + c * pt[:, 1]

    return pt_rot

def array_to_int_tuple(X):
    return (int(X[0]), int(X[1]))

def draw_str(dst, target, s, color = (0, 0, 0)):
    x, y = target
    cv2.putText(dst, s, (x+1, y+1), cv2.FONT_HERSHEY_PLAIN, 1.0, (color), thickness = 2, lineType=cv2.LINE_AA)
    cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)

def draw_keypoints(keypoints, im, color=(255, 0, 0), radius = 3, fill = 0):
    for k in keypoints:
        # radius = 3  # int(k.size / 2)
        center = (int(k[0]), int(k[1]))

        # Draw circle
        cv2.circle(im, center, radius, color, fill)

def draw_keypoints_by_number(keypoints, im, color = (255, 0, 0)):
    for k in keypoints:
        klass = str(int(k[2]))
        center = (int(k[0]), int(k[1]))
        cv2.putText(im, klass, center, cv2.FONT_HERSHEY_PLAIN, 0.8, color, thickness = 1, lineType=cv2.LINE_AA)
