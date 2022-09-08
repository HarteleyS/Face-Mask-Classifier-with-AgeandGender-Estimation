# Face-Mask-Classifier

IT IS RECOMMENDED TO OPEN THIS FILE ON GITHUB, BECAUSE THIS FILE INCLUDE IMAGES

All the files can be found on Google Drive: https://drive.google.com/drive/folders/1NPNusl3vAaxsbTx9I-TOlymqrUzEVU_8?usp=sharing

File Directories:
1. Application:
- ResearchProject/Face_mask_withAgeandGender.py (is the main code file for the application)

2. Dataset:
- ResearchProject/dataset (is the dataset we use to train our CNN models for Face-Mask Classifier)

3. Face Detector:
- ResearchProject/face_detector/deploy.prototxt (is the prototext file)
- ResearchProject/face_detector/res10_300x300_ssd_iter_140000.caffemodel (is the caffe model or weight file)

4. Face-Mask Classifier:
- ResearchProject/models/TrainingFile.ipnyb (is a notebook for the code of data preprocessing, hyperparameter tuning, and models training)
- ResearchProject/models/VGG16.h5 (is the weight of the VGG-16 model)
- ResearchProject/models/RESNET.h5 (is the weight of the ResNet-50 model)
- ResearchProject/models/MOBILENET.h5 (is the weight of the MobileNet_V2 model)
- ResearchProject/models/GOOGLENET.h5 (is the weight of the GoogleNet model)
- ResearchProject/models/NASNET.h5 (is the weight of the NasNet model, the current one use on the main application)

5. Age and Gender Predictor:
- ResearchProject/gad/opencv_face_detector.pbtxt (is the pbtext file)
- ResearchProject/gad/opencv_face_detector_uint8.pb (is the pb file)
- ResearchProject/gad/age_deploy.prototxt (is the age prototext file)
- ResearchProject/gad/age_net.caffemodel (is the age caffe model or weight file)
- ResearchProject/gad/gender_deploy.prototxt (is the gender prototext file)
- ResearchProject/gad/gender_net.caffemodel (is the gender caffe model or weight file)

Using the application:
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
1. importing libraries needed:
    - import os
    - import cv2
    - import imutils
    - import time
    - import numpy as np
    - from tensorflow.keras.preprocessing.image import img_to_array
    - from tensorflow.keras.models import load_model
    - from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    - from imutils.video import VideoStream
    - from imutils.video import FileVideoStream
    - import tkinter as tk
    - from tkinter import filedialog
    - from tkinter import *
    - import math
    - import argparse


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


Application Guide:
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
1. Open the Face_mask_withAgeandGender.py
2. Click the run button or F5 on keyboard
![Screenshot_66](https://user-images.githubusercontent.com/98985214/189163054-c95106ed-1101-4c5d-b840-0a1e5d688d40.png)

3. The User Interface will pop-up wtih 3 options
![Screenshot_65](https://user-images.githubusercontent.com/98985214/189163361-bd60e21f-e964-4374-b803-db2dae4a76d5.png)

4. Live Stream will be using your camera on your device
![Screenshot_63](https://user-images.githubusercontent.com/98985214/189163983-1033cf73-ac7d-45a6-92d3-54f0db8dc2d7.png)

5. Video Stream will let you choose the file from your folder on your device
![Screenshot_64](https://user-images.githubusercontent.com/98985214/189164271-aab3d6f3-bee6-4e1c-a937-83435a3d7172.png)

6. Exit button is for you to quit from the application




