import numpy as np
import pandas as pd
import random
import cv2
import tkinter as tk
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import re
import os
import dlib_functions
from math import *
import dlib
from scipy.spatial import distance
from imutils import face_utils

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("C:/Users/lewsz/OneDrive/Desktop/master_thesis/shape_predictor_68_face_landmarks.dat")

def Area(corners):
    n = len(corners) # of corners
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += corners[i][0] * corners[j][1]
        area -= corners[j][0] * corners[i][1]
    area = abs(area) / 2.0
    return area

def landmark_processor(image):
    rects = detector(image, 1)

    for (i, rect) in enumerate(rects):
        shape = predictor(image, rect)
        shape = face_utils.shape_to_np(shape)

    shape = shape.tolist()
    return shape

sample = random.sample(os.listdir("../../faces/high_quality_dataset/total_dataset"), 100)
sample = ["../../faces/high_quality_dataset/total_dataset/" + choice for choice in sample]

images = [cv2.imread(path) for path in sample]
images = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images]
images = [cv2.resize(image, (400,500)) for image in images]
print('done')

image_names = [re.search(r"image_\d+.jpg", path).group() for path in sample]

colnames = ['s1', 's2', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9',
                'a10', 'a11', 'a12', 'a13', 'a14', 'a15', 'a16', 'a17', 'a18', 'a19',
                'a20', 'a21', 'a22', 'a23','d1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8', 'd9',
                'd10', 'd11', 'd12', 'd13', 'd14', 'd15', 'd16', 'd17','da1', 'da2', 'da3']
final_dataset = pd.DataFrame(columns = colnames)

for image in images:
    shape = landmark_processor(image)
    parameters = dlib_functions.parameter_processor(shape)
    final_dataset = final_dataset.append(parameters, ignore_index=True)

final_dataset['image_name'] = image_names

final_dataset.to_csv("SCALED.csv", encoding = 'utf-8', index = False)