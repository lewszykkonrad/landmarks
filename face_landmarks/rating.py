from operator import imod
from re import sub
import cv2
import os
import math
import dlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imutils
import random
from imutils import face_utils
from scipy.spatial import distance

from PIL import Image
from IPython.display import display, Image

from shapely.geometry import Polygon

import subprocess

import PySimpleGUI as sg


choices = random.sample(os.listdir("../../faces/high_quality_dataset/hot"), 2)
choices = ["../../faces/high_quality_dataset/hot/" + choice for choice in choices]

images = [cv2.imread(file) for file in choices]
# for image in images:
#     # plt.imshow(image)
#     # plt.show()
#     p = subprocess.Popen(["display", image])
#     # cv2.imshow(image, cv2.IMREAD_ANYCOLOR)
#     input("do something")
#     p.kill
print(choices)
subprocess.Popen(["display", choices[1]])
# image = cv2.imread(choice)
# plt.imshow(image)
# plt.show()

sg.Window('My window').Layout([[ sg.Image(choices[1]) ]]).Read()

#display(Image(filename = "../../faces/high_quality_dataset/hot/" + choice))

