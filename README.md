# Face-Mask-Classifier

Using the application:
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
1. importing libraries needed:
        import os
        import cv2
        import imutils
        import time
        import numpy as np
        from tensorflow.keras.preprocessing.image import img_to_array
        from tensorflow.keras.models import load_model
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        from imutils.video import VideoStream
        from imutils.video import FileVideoStream
        import tkinter as tk
        from tkinter import filedialog
        from tkinter import *
        import math
        import argparse


2. Files needed for application to run:
    - "face_detector/deploy.prototxt"
    - "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
    - "models/NASNET.h5"
    - "gad/opencv_face_detector.pbtxt"
    - "gad/opencv_face_detector_uint8.pb"
    - "gad/age_deploy.prototxt"
    - "gad/age_net.caffemodel"
    - "gad/gender_deploy.prototxt"
    - "gad/gender_net.caffemodel"
