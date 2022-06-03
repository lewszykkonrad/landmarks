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


choice = random.choice(os.listdir("../../faces/high_quality_dataset/hot"))
choice = "../../faces/high_quality_dataset/hot/" + choice

image = cv2.imread(choice)
plt.imshow(image)
plt.show()


#display(Image(filename = "../../faces/high_quality_dataset/hot/" + choice))

