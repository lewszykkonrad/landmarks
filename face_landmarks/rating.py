import cv2
import os
import math
import dlib
import numpy as np
import pandas as pd
import imutils
import random
from imutils import face_utils
from scipy.spatial import distance
from PIL import Image, ImageTk
from shapely.geometry import Polygon
#package used for GUI
import tkinter as tk 


#list for storing the ratings user has given
training_sequence = []
test_sequence = []

sample = random.sample(os.listdir("../../faces/high_quality_dataset/hot"), 50)
sample = ["../../faces/high_quality_dataset/hot/" + choice for choice in sample]

images = [Image.open(x) for x in sample]
images = [x.resize((400,500)) for x in images]

training_images = images[:25]
testing_images = images[25:]

###################
# Training sample #
###################
window = tk.Tk()

frame_image = ImageTk.PhotoImage(training_images.pop())

frame_a = tk.Frame()
frame_b = tk.Frame()

def on_click(value):
    training_sequence.append(value)

def change_pic():
    if len(training_images) == 0:
        window.destroy()
    else:
        photo = ImageTk.PhotoImage(training_images.pop())
        picture_placeholder.configure(image=photo)
        picture_placeholder.image = photo 
    print("updated")

picture_placeholder = tk.Label(
    master = frame_a,
    image = frame_image,
    text = "Picture",
    foreground= "white",
    background="black"
    )
picture_placeholder.pack()

button_1 = tk.Button(
    master = frame_b, 
    command = lambda: [on_click(1), change_pic()],
    text = "1",
    width=25,
    height=5,
    foreground="white",
    background="black")
button_1.grid(row = 0, column=0)

button_2 = tk.Button(
    master = frame_b, 
    command = lambda: [on_click(2), change_pic()],
    text = "2",
    width=25,
    height=5,
    foreground="white",
    background="black")
button_2.grid(row = 0, column=1)

button_3 = tk.Button(
    master = frame_b, 
    command = lambda: [on_click(3), change_pic()],
    text = "3",
    width=25,
    height=5,
    foreground="white",
    background="black")
button_3.grid(row = 0, column=2)

frame_b.pack()
frame_a.pack()

window.mainloop()

##################
# testing sample #
##################


print(len(training_sequence))
print(training_sequence)

