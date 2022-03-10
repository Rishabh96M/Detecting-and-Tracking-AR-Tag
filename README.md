[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

Verified on python 3.8.10 and packages used are Random, NumPy, Matplotlib and cv2

# Detecting-and-Tracking-AR-Tag
This project will focus on detecting a custom AR Tag (a form offiducial marker), that is used for obtaining a point of reference in the real world, such as in augmented reality applications. There are two aspects to using an AR Tag, namely detection and tracking, both of which will be implemented in this project. The detection stage will involve finding the AR Tag from a given image sequence while the tracking stage will involve keeping the tag in “view” throughout the sequence and performing image processing operations based on the tag’s orientation and position (a.k.a. the pose).

## Steps to run
To clone the file:
```
git clone https://github.com/rishabh96m/Detecting-and-Tracking-AR-Tag.git
cd Detecting-and-Tracking-AR-Tag
```
To run the Detection:
```
python3 detection.py
```

To run the Tracking:
```
python3 tracking.py
```

## To install the dependencies
```
sudo pip install matplotlib
sudo pip install opencv-python
sudo pip install numpy
```
