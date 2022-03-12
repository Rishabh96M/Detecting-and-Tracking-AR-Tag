# Template projrction on AR Tag
# Copyright (c) 2022 Rishabh Mukund
# MIT License
#
# Description: Tracking the AR Tag in a video, and projecting a given template
# on it. Fft and homography are used to detect the tag and warping is used to
# project the template on the AR tag in the video.

import cv2
import numpy as np
import utils

if __name__ == '__main__':
    tag = np.zeros((80, 80), dtype=np.uint8)

    cap = cv2.VideoCapture('Resources/1tagvideo.mp4')
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            rframe = cv2.resize(frame, (960, 540))
            gray = cv2.cvtColor(rframe, cv2.COLOR_RGB2GRAY)
            img = utils.removeBackground(gray)
            edges = utils.edgeDetection(img)
            corners = utils.getCorners(edges)
            cv2.imshow('input', rframe)

            try:
                h = utils.homography(corners[:, 0], corners[:, 1], [
                                     0, 80, 80, 0], [0, 0, 80, 80])
                H = np.linalg.inv(h)
                tag = utils.inverseWarping(gray, h, tag)

                _, ori = utils.getARTagID(tag)
                cv2.imshow('tag', tag)

                template = cv2.imread('Resources/testudo.png')
                template = cv2.resize(template, (80, 80))
                for i in range(ori):
                    template = cv2.rotate(
                        template, cv2.ROTATE_90_CLOCKWISE)

                dst = utils.Warping(template, H, rframe)
                cv2.imshow('output', dst)

            except (TypeError, IndexError):
                continue

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()
