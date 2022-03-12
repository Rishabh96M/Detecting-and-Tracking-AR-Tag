[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

Verified on python 3.8.10 and packages used are Random, NumPy, Matplotlib and cv2

# Detecting-and-Tracking-AR-Tag
This project will focus on detecting a custom AR Tag (a form offiducial marker), that is used for obtaining a point of reference in the real world, such as in augmented reality applications. There are two aspects to using an AR Tag, namely detection and tracking, both of which will be implemented in this project. The detection stage will involve finding the AR Tag from a given image sequence while the tracking stage will involve keeping the tag in “view” throughout the sequence and performing image processing operations based on the tag’s orientation and position (a.k.a. the pose).

## Steps to run the program
To clone the files:
```
git clone https://github.com/rishabh96m/Detecting-and-Tracking-AR-Tag.git
```
To run the Detection code:
```
cd Detecting-and-Tracking-AR-Tag
python3 detection.py
```

To run the template projection code:
```
cd Detecting-and-Tracking-AR-Tag
python3 template_projection.py
```

To run the cube projection code:
```
cd Detecting-and-Tracking-AR-Tag
python3 cube_projection.py
```

## To install the dependencies
```
sudo pip install matplotlib
sudo pip install opencv-python
sudo pip install numpy
```

The ***Resources*** folder contains the template to be imposed and the video of the AR tag.
The ***Videos*** folder contains the videos of the code output for cube projection and template projection
