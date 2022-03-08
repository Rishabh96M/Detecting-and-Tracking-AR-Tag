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
            img = utils.removeBackground(gray)
            edges = utils.edgeDetection(img)
            corners = utils.getCorners(edges)

            for point in corners:
                cv2.circle(frame, point, 8, [255, 0, 0], -1)

            try:
                H = utils.homography(corners[:, 0], corners[:, 1], [
                                     0, 80, 80, 0], [0, 0, 80, 80])

                tag = utils.inverseWarping(gray, H, (80, 80))

                id, ori = utils.getARTagID(tag)

                template = cv2.imread('Resources/testudo.png')
                template = cv2.resize(template, (80, 80))
                for i in range(ori):
                    template = cv2.rotate(
                        template, cv2.ROTATE_90_CLOCKWISE)

                dst = utils.fwdWarping(template, np.linalg.inv(H), frame)
                cv2.imshow('dst', dst)
            except (TypeError, IndexError):
                continue

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            break
    cap.release()
    cv2.destroyAllWindows()
