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
import mediapipe

user_query = input("what image would you like to process? (what is the number of the image?):  ")

path = "../../faces/high_quality_dataset/hot/image_" + user_query + ".jpg"

img_base = cv2.imread(path)
img = img_base.copy()

faceModule = mediapipe.solutions.face_mesh

face_mesh = faceModule.FaceMesh(static_image_mode=True)
results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

landmarks = results.multi_face_landmarks[0]

facial_areas = {
    'Contours': faceModule.FACEMESH_CONTOURS
    , 'Lips': faceModule.FACEMESH_LIPS
    , 'Face_oval': faceModule.FACEMESH_FACE_OVAL
    , 'Left_eye': faceModule.FACEMESH_LEFT_EYE
    , 'Left_eye_brow': faceModule.FACEMESH_LEFT_EYEBROW
    , 'Right_eye': faceModule.FACEMESH_RIGHT_EYE
    , 'Right_eye_brow': faceModule.FACEMESH_RIGHT_EYEBROW
    , 'Tesselation': faceModule.FACEMESH_TESSELATION
}

def plot_landmark(img_base, facial_area_name, facial_area_obj):
    
    print(facial_area_name, ":")
    
    img = img_base.copy()
    
    for source_idx, target_idx in facial_area_obj:
        source = landmarks.landmark[source_idx]
        target = landmarks.landmark[target_idx]

        relative_source = (int(img.shape[1] * source.x), int(img.shape[0] * source.y))
        relative_target = (int(img.shape[1] * target.x), int(img.shape[0] * target.y))

        cv2.line(img, relative_source, relative_target, (255, 255, 255), thickness = 2)
    
    fig = plt.figure(figsize = (15, 15))
    plt.axis('off')
    plt.imshow(img[:, :, ::-1])
    plt.show()


facial_area_obj = facial_areas["Left_eye_brow"]

plot_landmark(img_base, "left eyebrow", facial_area_obj)