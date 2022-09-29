from email.utils import encode_rfc2231
from turtle import end_fill
import cv2
import os
import math
import dlib
import numpy as np
import pandas as pd
import imutils
import random
import re
from imutils import face_utils
from scipy.spatial import distance
from PIL import Image, ImageTk
from shapely.geometry import Polygon
#package used for GUI
import tkinter as tk

from sqlalchemy import between 
#the dlib functions I wrote in dlib_functions.py
import dlib_functions


#list for storing the ratings user has given
sequence = []

#the sample of paths for images
sample = random.sample(os.listdir("../../faces/high_quality_dataset/hot"), 10)
sample = ["../../faces/high_quality_dataset/hot/" + choice for choice in sample]

#it will be useful to keep the names of the pictures that the user is rating. Additionally, it might prove useful for future comparisons 
#between users
image_names = [re.search(r"image_\d+.jpg", path).group() for path in sample]

images = [Image.open(x) for x in sample]
images = [x.resize((400,500)) for x in images]


#the  function I wrote in dlib_functions uses path as an input, while here I have processed images meant for display to the user. I will
#save the paths of the images and later pass them into the dlib_function

#appending values on binary and trinary ratings
def on_click(value):
    sequence.append(value)

def change_pic():
    if len(images) == 0:
        window.destroy()
    else:
        photo = ImageTk.PhotoImage(images.pop(0))
        picture_placeholder.configure(image=photo)
        picture_placeholder.image = photo 
    print("updated")

###################
# Training sample #
###################
def user_selects_rating_choice():
    choice = input("how would you like to rate the faces? (choices are: binary, trinary, rational) ")
    if choice not in ["binary", "trinary", "rational"]:
        print("please select one of the appropriate options")
        return user_selects_rating_choice()
    else:
        return choice

rating_choice = user_selects_rating_choice()


###################
#    BINARY       #
###################
def binary():
    window = tk.Tk()
    frame_image = ImageTk.PhotoImage(images.pop(0))

    frame_a = tk.Frame()
    frame_b = tk.Frame()

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

    frame_b.pack()
    frame_a.pack()
    window.mainloop()

    #initiating a dataframe for storing rating results
    colnames = ['s1', 's2', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9',
                    'a10', 'a11', 'a12', 'a13', 'a14', 'a15', 'a16', 'a17', 'a18', 'a19',
                    'a20', 'a21', 'a22', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8', 'd9',
                    'd10', 'd11', 'd12', 'd13', 'd14', 'da1', 'da2']
    final_dataset = pd.DataFrame(columns = colnames)

    for path in sample:
        shape = dlib_functions.facial_landmark_processor(path)
        parameters = dlib_functions.parameter_processor(shape)
        final_dataset = final_dataset.append(parameters, ignore_index=True)

    final_dataset['image_name'] = image_names
    final_dataset['rating'] = sequence
    return(final_dataset)

###################
#    TRINARY      #
###################
def trinary():
    window = tk.Tk()
    frame_image = ImageTk.PhotoImage(images.pop(0))

    frame_a = tk.Frame()
    frame_b = tk.Frame()

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

    #initiating a dataframe for storing rating results
    colnames = ['s1', 's2', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9',
                    'a10', 'a11', 'a12', 'a13', 'a14', 'a15', 'a16', 'a17', 'a18', 'a19',
                    'a20', 'a21', 'a22', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8', 'd9',
                    'd10', 'd11', 'd12', 'd13', 'd14', 'da1', 'da2']
    final_dataset = pd.DataFrame(columns = colnames)

    for path in sample:
        shape = dlib_functions.facial_landmark_processor(path)
        parameters = dlib_functions.parameter_processor(shape)
        final_dataset = final_dataset.append(parameters, ignore_index=True)

    final_dataset['image_name'] = image_names
    final_dataset['rating'] = sequence
    return(final_dataset)

###################
#    RATIONAL      #
###################
def rationaL_rating():
    sequence = []

    sample = random.sample(os.listdir("../../faces/high_quality_dataset/hot"), 5)
    sample = ["../../faces/high_quality_dataset/hot/" + choice for choice in sample]

    image_names = [re.search(r"image_\d+.jpg", path).group() for path in sample]

    images = [Image.open(x) for x in sample]
    images = [x.resize((400,500)) for x in images]

    def change_picture_and_store_rating(event): 
        if 1 <= eval(entry.get()) <= 5:
            res.configure(text = "Result: " + str(eval(entry.get())))
            sequence.append(eval(entry.get()))
            if len(images) == 0:
                window.destroy()
            else:
                photo = ImageTk.PhotoImage(images.pop(0))
                picture_placeholder.configure(image=photo)
                picture_placeholder.image = photo 
        
    window = tk.Tk()
    frame_a = tk.Frame()
    frame_image = ImageTk.PhotoImage(images.pop(0))

    picture_placeholder = tk.Label(
        master = frame_a,
        image = frame_image,
        text = "Picture",
        foreground= "white",
        background="black"
        )
    picture_placeholder.pack()

    tk.Label(window, text="You are rating in rational numbers from 1 to 5 (you can input digits after the comma):").pack()

    entry = tk.Entry(window)
    entry.bind("<Return>", change_picture_and_store_rating)

    frame_a.pack()
    entry.pack()
    res = tk.Label(window)
    res.pack()

    window.mainloop()

    #initiating a dataframe for storing rating results
    colnames = ['s1', 's2', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9',
                    'a10', 'a11', 'a12', 'a13', 'a14', 'a15', 'a16', 'a17', 'a18', 'a19',
                    'a20', 'a21', 'a22', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8', 'd9',
                    'd10', 'd11', 'd12', 'd13', 'd14', 'da1', 'da2']

    final_dataset = pd.DataFrame(columns = colnames)

    for path in sample:
        shape = dlib_functions.facial_landmark_processor(path)
        parameters = dlib_functions.parameter_processor(shape)
        final_dataset = final_dataset.append(parameters, ignore_index=True)

    final_dataset['image_name'] = image_names
    final_dataset['rating'] = sequence

    return(final_dataset)



filename = input("please enter your first and last name in this format: firstname_lastname  ")
filename = "datasets/" + filename + ".csv"

#final_dataset.to_csv(filename, encoding = 'utf-8', index = False)
