# AR Tag Detection
# Copyright (c) 2022 Rishabh Mukund
# MIT License
#
# Description: Detecting AR Tag in a given frame, check for orientation and
# calculate thr ID of the tag using fft and homography.

import cv2
import utils

if __name__ == '__main__':
    cap = cv2.VideoCapture('Resources/1tagvideo.mp4')

    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    edges = utils.edgeDetection(gray)
    corners = utils.getCorners(edges)

    for point in corners:
        cv2.circle(frame, point, 8, [255, 0, 0], -1)
    cv2.imshow('frame', frame)

    H = utils.homography(corners[:, 0], corners[:, 1], [
                         0, 80, 80, 0], [0, 0, 80, 80])

    tag = utils.inverseWarping(gray, H, (80, 80))
    cv2.imshow('tag', tag)

    id = utils.getARTagID(tag)
    print(id)

    cv2.waitKey(0)
    cap.release()
    cv2.destroyAllWindows()
