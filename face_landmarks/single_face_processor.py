#This code will be helpful in occasional checking for consistency. I will be able to quickly process a given face and make sure
#that my databases are correct, and the parameters/landmarks are correctly intepreted
import cv2
import os
import math
import dlib
from matplotlib.style import use
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imutils
from imutils import face_utils
from scipy.spatial import distance

from shapely.geometry import Polygon

#detector and predictor needed for the dlib landmark detection (detector detects the face, predictor predicts the landmarks)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("C:/Users/lewsz/OneDrive/Desktop/master_thesis/shape_predictor_68_face_landmarks.dat")

#Specifying text and color of marks on the face
font = cv2.FONT_HERSHEY_SIMPLEX
org = (236, 135)
fontScale = 0.2
color = (255, 255, 255)
thickness = 1

user_query = input("what image would you like to process? (what is the number of the image?):  ")

path = "../../faces/high_quality_dataset/hot/image_" + user_query + ".jpg"
image = cv2.imread(path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

rects = detector(image, 1)

for (i, rect) in enumerate(rects):

    shape = predictor(image, rect)
    shape = face_utils.shape_to_np(shape)

    index = 0
    for (x, y) in shape:
        cv2.circle(image, (x, y), 1, (0, 255, 0), 3)
        cv2.putText(image, str(index), (x, y), font, 
                    fontScale, color, thickness, cv2.LINE_AA)
        index += 1

    
cv2.imshow("image window",image)
cv2.waitKey(0)