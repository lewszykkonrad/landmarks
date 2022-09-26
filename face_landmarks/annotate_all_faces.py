import cv2
import os
import math
import dlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imutils
from imutils import face_utils
from scipy.spatial import distance
import dlib_functions

#detector and predictor needed for processing the face imagy by dlib and then predicting the landmarks on that image
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("C:/Users/lewsz/OneDrive/Desktop/master_thesis/shape_predictor_68_face_landmarks_GTX.dat")

#folder with all the faces
faces_folder_path = "C:/Users/lewsz/OneDrive/Desktop/master_thesis/faces/high_quality_dataset/total_dataset"
face_files = os.listdir(faces_folder_path)

#target folder where annotated faces will be stored
target_path = "annotated_faces/"

#Specifying text and color of marks on the face
font = cv2.FONT_HERSHEY_SIMPLEX
org = (236, 135)
fontScale = 0.2
color = (255, 255, 255)
thickness = 1

#annotating landmarks on each image
for file in face_files:

    path = faces_folder_path + "/" + file
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #rects are the coordinates of each landmark, and detector is the dlib model that finds them on the face
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
    
    cv2.imwrite(os.path.join(target_path , file), image)

