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
    k = np.array([[1346.1, 0, 932.16],
                  [0, 1355.93, 654.9],
                  [0, 0, 1]])
    K = np.linalg.inv(k)

    cap = cv2.VideoCapture('Resources/1tagvideo.mp4')
    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            img = utils.removeBackground(gray)
            edges = utils.edgeDetection(img)
            corners = utils.getCorners(edges)

            try:
                h = utils.homography(corners[:, 0], corners[:, 1], [
                                     0, 80, 80, 0], [0, 0, 80, 80])
                H = np.linalg.inv(h)
                tag = utils.inverseWarping(gray, h, (80, 80))

                id, ori = utils.getARTagID(tag)

                template = cv2.imread('Resources/testudo.png')
                template = cv2.resize(template, (80, 80))
                for i in range(ori):
                    template = cv2.rotate(
                        template, cv2.ROTATE_90_CLOCKWISE)

                dst = utils.fwdWarping(template, H, frame)

                P = utils.getProjMat(k, K, H)

                cube_points = np.array([[0, 0, 0, 1], [0, 80, 0, 1],
                                        [80, 80, 0, 1], [80, 0, 0, 1],
                                        [0, 0, -80, 1], [0, 80, -80, 1],
                                        [80, 80, -80, 1], [80, 0, -80, 1]])

                for point in cube_points:
                    temp = np.matmul(P, point)
                    temp /= temp[-1]
                    cv2.circle(frame, (int(temp[1]), int(
                        temp[0])), 8, [0, 0, 255], -1)

                cv2.imshow('tracking', frame)
            except (TypeError, IndexError):
                continue

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()
