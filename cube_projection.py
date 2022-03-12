# Cube projrction on AR Tag
# Copyright (c) 2022 Rishabh Mukund
# MIT License
#
# Description: Tracking the AR Tag in a video, and projecting a cube on it,
# fft and homography are used to detect the tag and warping is used to project
# the cube

import cv2
import numpy as np
import utils

if __name__ == '__main__':
    k = np.array([[1346.1, 0, 932.16],
                  [0, 1355.93, 654.9],
                  [0, 0, 1]])
    K = np.linalg.inv(k)
    tag = np.zeros((80, 80), dtype=np.uint8)

    cap = cv2.VideoCapture('Resources/1tagvideo.mp4')
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            img = utils.removeBackground(gray)
            edges = utils.edgeDetection(img)
            corners = utils.getCorners(edges)
            cv2.imshow('input', frame)

            try:
                h = utils.homography(corners[:, 0], corners[:, 1], [
                                     0, 80, 80, 0], [0, 0, 80, 80])
                H = np.linalg.inv(h)
                tag = utils.inverseWarping(gray, h, tag)

                _, ori = utils.getARTagID(tag)
                cv2.imshow('tag', tag)

                P = utils.getProjMat(k, K, H)
                P = P/P[-1, -1]

                cube_points = np.array([[0, 0, 0, 1], [0, 80, 0, 1],
                                        [80, 80, 0, 1], [80, 0, 0, 1],
                                        [0, 0, -80, 1], [0, 80, -80, 1],
                                        [80, 80, -80, 1], [80, 0, -80, 1]])

                new_points = np.matmul(P, cube_points.T)
                new_points /= new_points[-1]
                new_points = np.int0(new_points.T)
                for i in range(3):
                    cv2.line(frame, new_points[i, 0:2],
                             new_points[i+1, 0:2], [255, 0, 0], 4)
                    cv2.line(frame, new_points[i+4, 0:2],
                             new_points[i+5, 0:2], [255, 0, 0], 4)
                    cv2.line(frame, new_points[i, 0:2],
                             new_points[i+4, 0:2], [255, 0, 0], 4)
                cv2.line(frame, new_points[3, 0:2],
                         new_points[0, 0:2], [255, 0, 0], 4)
                cv2.line(frame, new_points[7, 0:2],
                         new_points[4, 0:2], [255, 0, 0], 4)
                cv2.line(frame, new_points[3, 0:2],
                         new_points[7, 0:2], [255, 0, 0], 4)

                cv2.imshow('tracking', frame)
            except (TypeError, IndexError):
                continue

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()
