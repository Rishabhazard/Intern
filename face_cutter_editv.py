# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 17:16:10 2020

@author: rishu
"""

# USAGE
# python face_cutter_editv.py


# import the necessary packages
from imutils import paths
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os

# construct the argument parser and parse the arguments
def readable_dir(prospective_dir):
  if not os.path.isdir(prospective_dir):
    raise Exception("readable_dir:{0} is not a valid path".format(prospective_dir))
  if os.access(prospective_dir, os.R_OK):
    return prospective_dir
  else:
    raise Exception("readable_dir:{0} is not a readable dir".format(prospective_dir))

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", type=readable_dir, default='dataset',
	help="path to input directory of faces + images")

ap.add_argument("-d", "--detector", type=readable_dir, default='face_detection_model',
	help="path to OpenCV's deep learning face detector")

ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)


# grab the paths to the input images in our dataset
print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(args["dataset"]))


# initialize the total number of faces processed
total = 0

# loop over the image paths
temp = 0;
for (i, imagePath) in enumerate(imagePaths):
    
	# extract the person name from the image path
    print("[INFO] processing image {}/{}".format(i + 1,
        len(imagePaths)))
    name = imagePath.split(os.path.sep)[-2]

	# load the image, resize it to have a width of 600 pixels (while
	# maintaining the aspect ratio), and then grab the image
	# dimensions
    image = cv2.imread(imagePath)
    image = imutils.resize(image, width=600)
    (h, w) = image.shape[:2]
    

	# construct a blob from the image
    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False)

	# apply OpenCV's deep learning-based face detector to localize
	# faces in the input image
    detector.setInput(imageBlob)
    detections = detector.forward()

	# ensure at least one face was found
    if len(detections) > 0:
		# we're making the assumption that each image has only ONE
		# face, so find the bounding box with the largest probability
        i = np.argmax(detections[0, 0, :, 2])
        confidence = detections[0, 0, i, 2]

		# ensure that the detection with the largest probability also
		# means our minimum probability test (thus helping filter out
		# weak detections)
        if confidence > args["confidence"]:
            # compute the (x, y)-coordinates of the bounding box for
			# the face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

			# extract the face ROI and grab the ROI dimensions
            face = image[(startY-75):(endY+75), (startX-125):(endX+125)]
            
            (fH, fW) = face.shape[:2]
            
			# ensure the face width and height are sufficiently large
            if fW < 20 or fH < 20:
                continue
            # output of the cropped images 
            cv2.imwrite('{}.jpg'.format(temp),face)
            temp += 1
