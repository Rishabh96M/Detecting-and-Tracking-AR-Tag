# AR Tag Tracking
# Copyright (c) 2022 Rishabh Mukund
# MIT License
#
# Description: Tracking the AR Tag in a video, check for orientation and
# calculate thr ID of the tag using fft and homography.

import cv2
import numpy as np
import utils

if __name__ == '__main__':
    cap = cv2.VideoCapture('Resources/1tagvideo.mp4')

    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
            close = cv2.morphologyEx(
                thresh, cv2.MORPH_CLOSE, np.ones((101, 101)))
            open = cv2.morphologyEx(
                close, cv2.MORPH_OPEN, np.ones((11, 11)))
            blurred = cv2.GaussianBlur(open, (9, 9), 0)
            points = np.int0(cv2.goodFeaturesToTrack(blurred, 4, 0.1, 200))

            corners = utils.find_square(points[:, 0], 50)
            for point in corners:
                cv2.circle(frame, point, 8, [255, 0, 0], -1)
            cv2.imshow('frame', frame)

            # H = utils.homography(corners[:, 0], corners[:, 1], [
            #                      0, 80, 80, 0], [0, 0, 80, 80])
            #
            # tag = utils.inverseWarping(gray, H, (80, 80))
            # cv2.imshow('tag', tag)
            #
            # id = utils.getARTagID(tag)
            # print(id)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        else:
            break

    cap.release()
    cv2.destroyAllWindows()
