#!/bin/bash
g++ -I/usr/local/include/opencv -I/usr/local/include/opencv2 -L/usr/local/lib/ -g -o OpenCV  OpenCV.cpp -lopencv_core -lopencv_videoio -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc
