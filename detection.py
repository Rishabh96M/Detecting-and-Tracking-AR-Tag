# AR Tag Detection
# Copyright (c) 2022 Rishabh Mukund
# MIT License
#
# Description: Detecting AR Tag in a given frame, check for orientation and
# calculate thr ID of the tag using fft and homography.

import cv2
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    cap = cv2.VideoCapture('Resources/1tagvideo.mp4')

    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            cv2.imshow('Gray Image', gray)
            _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
            cv2.imshow('Thresholding', thresh)

            close = cv2.morphologyEx(
                thresh, cv2.MORPH_CLOSE, np.ones((101, 101)))
            cv2.imshow('close', close)

            open = cv2.morphologyEx(
                close, cv2.MORPH_OPEN, np.ones((11, 11)))
            cv2.imshow('open', open)

            blurred = cv2.GaussianBlur(open, (9, 9), 0)
            cv2.imshow('blur', blurred)

            corners = cv2.goodFeaturesToTrack(blurred, 4, 0.1, 200)
            corners = np.int0(corners)

            for i in corners:
                x, y = i.ravel()
                cv2.circle(frame, (x, y), 8, [0, 0, 255], -1)

            cv2.imshow('frame', frame)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        else:
            break

    cap.release()
    cv2.destroyAllWindows()
