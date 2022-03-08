# utils.py>
#
# AR Tag Utils
# Copyright (c) 2022 Rishabh Mukund
# MIT License
#
# Description: Functions definitions to Track and Detect AR Tag

import cv2
import numpy as np
from itertools import combinations


def removeBackground(gray):
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    _, ithresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    close = cv2.morphologyEx(
        thresh, cv2.MORPH_CLOSE, np.ones((101, 101)))
    _, mask = cv2.threshold(close, 127, 1, cv2.THRESH_BINARY)
    open = cv2.morphologyEx(
        ithresh * mask, cv2.MORPH_OPEN, np.ones((5, 5)))
    close = cv2.morphologyEx(
        open, cv2.MORPH_CLOSE, np.ones((71, 71)))
    return close


def edgeDetection(close):
    blurred = cv2.GaussianBlur(close, (7, 7), 0)

    dft = cv2.dft(np.float32(blurred), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    rows, cols = close.shape
    crow, ccol = int(rows / 2), int(cols / 2)
    mask = np.ones((rows, cols, 2), np.uint8)
    r = 100
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
    mask[mask_area] = 0

    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    return cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])


def find_square(pts_list, thresh=5):
    for pts in combinations(pts_list, 4):
        pts = np.array(pts)
        d01 = dist(pts[0], pts[1])
        d02 = dist(pts[0], pts[2])
        d03 = dist(pts[0], pts[3])
        d21 = dist(pts[2], pts[1])
        d31 = dist(pts[1], pts[3])
        d32 = dist(pts[2], pts[3])
        flag = False

        if(abs(d01 - d32) < thresh) and (abs(d31 - d02) < thresh):
            flag = True
        elif(abs(d02 - d31) < thresh) and (abs(d03 - d21) < thresh):
            flag = True
        elif(abs(d01 - d32) < thresh) and (abs(d03 - d21) < thresh):
            flag = True

        if flag:
            points = np.zeros((4, 2))
            idx = pts[:, 0].argsort()
            pts = pts[idx]
            points[0] = pts[0]
            points[2] = pts[-1]
            pts = np.delete(pts, [0, 3], axis=0)
            idx = pts[:, 1].argsort()
            pts = pts[idx]
            points[1] = pts[0]
            points[3] = pts[-1]
            return np.int0(points)
    return []


def getCorners(edges):
    open = cv2.morphologyEx(edges, cv2.MORPH_OPEN, np.ones((3, 3)))
    points = np.int0(cv2.goodFeaturesToTrack(open, 8, 0.1, 70))
    corners = find_square(points[:, 0], thresh=20)
    return corners


def dist(point1, point2):
    return(abs((point1[0] - point2[0]) + (point1[1] - point2[1])))


def checkIfTag(points):
    grid_x = np.int8(np.linspace(points[0], points[1], 9))
    grid_y = np.int8(np.linspace(points[1], points[2], 9))
    return grid_x, grid_y


def homography(x, y, xp, yp):
    A = np.array([[-x[0], -y[0], -1, 0, 0, 0, x[0]*xp[0], y[0]*xp[0], xp[0]],
                  [0, 0, 0, -x[0], -y[0], -1, x[0]*yp[0], y[0]*yp[0], yp[0]],
                  [-x[1], -y[1], -1, 0, 0, 0, x[1]*xp[1], y[1]*xp[1], xp[1]],
                  [0, 0, 0, -x[1], -y[1], -1, x[1]*yp[1], y[1]*yp[1], yp[1]],
                  [-x[2], -y[2], -1, 0, 0, 0, x[2]*xp[2], y[2]*xp[2], xp[2]],
                  [0, 0, 0, -x[2], -y[2], -1, x[2]*yp[2], y[2]*yp[2], yp[2]],
                  [-x[3], -y[3], -1, 0, 0, 0, x[3]*xp[3], y[3]*xp[3], xp[3]],
                  [0, 0, 0, -x[3], -y[3], -1, x[3]*yp[3], y[3]*yp[3], yp[3]]])

    _, _, VT = np.linalg.svd(A)
    return VT[-1].reshape((3, 3)) / VT[-1, -1]


def inverseWarping(src, H, dstSize):
    warped = np.zeros(dstSize, dtype=np.uint8)
    for x in range(dstSize[0]):
        for y in range(dstSize[1]):
            temp = np.matmul(np.linalg.inv(H), [x, y, 1])
            warped[y, x] = src[int(temp[1]/temp[-1]), int(temp[0]/temp[-1])]
    return warped


def fwdWarping(src, H, dst):
    for x in range(np.shape(src)[0]):
        for y in range(np.shape(src)[1]):
            temp = np.matmul(H, [x, y, 1])
            dst[int(temp[1]/temp[-1]), int(temp[0]/temp[-1])] = src[y, x]
    return dst


def getARTagID(img):
    _, thresh = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)
    x = np.int8(np.linspace(0, np.shape(img)[0], 9))
    y = np.int8(np.linspace(0, np.shape(img)[1], 9))

    border_mask = np.ones(np.shape(img))
    border_mask[y[2]:y[6], x[2]:x[6]] = 0
    border_mask = border_mask / np.sum(border_mask)
    if not np.sum(thresh * border_mask) < 0.1:
        return -1, _

    k = np.ones((x[1], y[1])) / 100

    if np.sum(thresh[y[5]:y[6], x[5]:x[6]] * k) > 0.9:
        order = [1, 2, 4, 8]
        ori = 0
    elif np.sum(thresh[y[5]:y[6], x[2]:x[3]] * k) > 0.9:
        order = [8, 1, 2, 4]
        ori = 1
    elif np.sum(thresh[y[2]:y[3], x[2]:x[3]] * k) > 0.9:
        order = [4, 8, 1, 2]
        ori = 2
    elif np.sum(thresh[y[2]:y[3], x[5]:x[6]] * k) > 0.9:
        order = [2, 4, 8, 1]
        ori = 3
    else:
        return -1, -1

    id = 0
    if np.sum(thresh[y[3]:y[4], x[3]:x[4]] * k) > 0.9:
        id += order[0]
    if np.sum(thresh[y[3]:y[4], x[4]:x[5]] * k) > 0.9:
        id += order[1]
    if np.sum(thresh[y[4]:y[5], x[4]:x[5]] * k) > 0.9:
        id += order[2]
    if np.sum(thresh[y[4]:y[5], x[3]:x[4]] * k) > 0.9:
        id += order[3]
    return id, ori
