# AR Tag Detection
# Copyright (c) 2022 Rishabh Mukund
# MIT License
#
# Description: Detecting AR Tag in a given frame, check for orientation and
# calculate thr ID of the tag using fft and homography.

import cv2
import utils
import numpy as np

if __name__ == '__main__':
    cap = cv2.VideoCapture('Resources/1tagvideo.mp4')
    _, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    img = utils.removeBackground(gray)
    edges = utils.edgeDetection(img)
    corners = utils.getCorners(edges)

    print("AR tag is at:")
    print(corners)

    for point in corners:
        cv2.circle(frame, point, 8, [255, 0, 0], -1)
    cv2.imshow('frame', frame)

    H = utils.homography(corners[:, 0], corners[:, 1], [
                         0, 80, 80, 0], [0, 0, 80, 80])

    tag = utils.inverseWarping(gray, H, np.zeros((80, 80), dtype=np.uint8))
    cv2.imshow('tag', tag)

    id, _ = utils.getARTagID(tag)
    print("\nID of AR tag is: ", id)

    cv2.waitKey(0)
    cap.release()
    cv2.destroyAllWindows()
