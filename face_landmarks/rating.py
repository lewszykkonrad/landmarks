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

import tkinter as tk

window = tk.Tk()

collected_values = []

# def button_clicked():
#     value = 

# def increase():
#     value = int(lbl_value["text"])
#     lbl_value["text"] = f"{value + 1}"

# frame_picture = tk.Frame()
# frame_rating_buttons = tk.Frame()


# lbl_value = tk.Label(master=frame_rating_buttons, text="0")
# lbl_value.grid(row=0, column=1)

# btn_1 = tk.Button(master=frame_rating_buttons, text="1", command=increase)
# btn_1.grid(row=0, column=0, sticky="nsew")

# btn_2 = tk.Button(master=frame_rating_buttons, text="2", command=increase)
# btn_2.grid(row=0, column=1, sticky="nsew")

# btn_3 = tk.Button(master=frame_rating_buttons, text="3", command=increase)
# btn_3.grid(row=0, column=2, sticky="nsew")

# btn_4 = tk.Button(master=frame_rating_buttons, text="4", command=increase)
# btn_4.grid(row=0, column=3, sticky="nsew")

# btn_5 = tk.Button(master=frame_rating_buttons, text="5", command=increase)
# btn_5.grid(row=0, column=4, sticky="nsew")

# frame_picture.pack()
# frame_rating_buttons.pack()

# window.mainloop()
#choosing random pictures from a chosen folder
choices = random.sample(os.listdir("../../faces/high_quality_dataset/hot"), 2)
choices = ["../../faces/high_quality_dataset/hot/" + choice for choice in choices]


# images = [cv2.imread(file) for file in choices]
# # for image in images:
# #     # plt.imshow(image)
# #     # plt.show()
# #     p = subprocess.Popen(["display", image])
# #     # cv2.imshow(image, cv2.IMREAD_ANYCOLOR)
# #     input("do something")
# #     p.kill
# print(choices)



