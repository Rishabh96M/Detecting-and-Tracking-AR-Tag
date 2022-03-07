# AR Tag Detection
# Copyright (c) 2022 Rishabh Mukund
# MIT License
#
# Description: Detecting AR Tag in a given frame, check for orientation and
# calculate thr ID of the tag using fft and homography.

import cv2
import numpy as np
from itertools import combinations


def dist(point1, point2):
    return(abs((point1[0] - point2[0]) + (point1[1] - point2[1])))


def find_square(pts_list, thresh=5):
    for pts in combinations(pts_list, 4):
        d01 = dist(pts[0], pts[1])
        d02 = dist(pts[0], pts[2])
        d03 = dist(pts[0], pts[3])
        d21 = dist(pts[2], pts[1])
        d31 = dist(pts[1], pts[3])
        d32 = dist(pts[2], pts[3])

        if(abs(d01 - d32) < thresh) and (abs(d31 - d02) < thresh):
            return [pts[0], pts[1], pts[3], pts[2]]

        if(abs(d02 - d31) < thresh) and (abs(d03 - d21) < thresh):
            return [pts[0], pts[2], pts[1], pts[3]]

        if(abs(d01 - d32) < thresh) and (abs(d03 - d21) < thresh):
            return [pts[0], pts[1], pts[2], pts[3]]
    return []


if __name__ == '__main__':
    cap = cv2.VideoCapture('Resources/1tagvideo.mp4')
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    dft = cv2.dft(np.float32(blurred), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    rows, cols = gray.shape
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
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    open = cv2.morphologyEx(
                img_back, cv2.MORPH_OPEN, np.ones((3, 3)))

    close = cv2.morphologyEx(
                open, cv2.MORPH_CLOSE, np.ones((3, 3)))

    corners = cv2.goodFeaturesToTrack(open, 8, 0.1, 70)
    corners = np.int0(corners)
    points = find_square(corners[:, 0])
    print(points)
    for point in points:
        cv2.circle(frame, point, 8, [255, 0, 0], -1)
    cv2.imshow('frame', frame)

    cv2.waitKey(0)
    cap.release()
    cv2.destroyAllWindows()
