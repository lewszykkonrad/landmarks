from email.utils import encode_rfc2231
from secrets import choice
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
from platformdirs import user_cache_dir
from scipy.spatial import distance
from PIL import Image, ImageTk
from shapely.geometry import Polygon
#package used for GUI
import tkinter as tk

from sqlalchemy import between 
#the dlib functions I wrote in dlib_functions.py
import dlib_functions




              
###################
# Training sample #
###################
def user_selects_rating_choice():
    choice = input("how would you like to rate the faces? (choices are: binary, quintary, rational) ")
    if choice not in ["binary", "quintary", "rational"]:
        print("please select one of the appropriate options")
        return user_selects_rating_choice()
    else:
        return choice

rating_choice = user_selects_rating_choice()


###################
#    BINARY       #
###################
def binary():
    
    global image_index
    image_index = 0
    sequence = []

    #the sample of paths for images
    sample = random.sample(os.listdir("../../faces/high_quality_dataset/hot"), 10)
    sample = ["../../faces/high_quality_dataset/hot/" + choice for choice in sample]

    #it will be useful to keep the names of the pictures that the user is rating. Additionally, it might prove useful for future comparisons 
    #between users
    image_names = [re.search(r"image_\d+.jpg", path).group() for path in sample]

    images = [Image.open(x) for x in sample]
    images = [x.resize((400,500)) for x in images]

    def on_click(value):
        sequence.append(value)

    def change_pic(): 
        global image_index
        image_index += 1
        if image_index == len(images):
            window.destroy()
        else:
            photo = ImageTk.PhotoImage(images[image_index])
            picture_placeholder.configure(image=photo)
            picture_placeholder.image = photo 

            image_filename = image_names[image_index]
            filename_placeholder.configure(text = image_filename)
            filename_placeholder.text = image_filename

    def go_back():
        if len(sequence) == 0:
                    return
        global image_index
        image_index -= 1
        sequence.pop()

        photo = ImageTk.PhotoImage(images[image_index])
        picture_placeholder.configure(image=photo)
        picture_placeholder.image = photo 

        image_filename = image_names[image_index]
        filename_placeholder.configure(text = image_filename)
        filename_placeholder.text = image_filename

    window = tk.Tk()
    frame_image = ImageTk.PhotoImage(images[image_index])

    frame_a = tk.Frame()
    frame_b = tk.Frame()
    frame_c = tk.Frame()
    frame_d = tk.Frame()

    picture_placeholder = tk.Label(
        master = frame_a,
        image = frame_image,
        text = "Picture",
        foreground= "white",
        background="black"
        )
    picture_placeholder.pack()

    filename_placeholder = tk.Label(
        master = frame_d,
        text = image_names[image_index],
        foreground= "black",
        background= "white",
        height= 5,
        width = 50,
        font = 10
    )
    filename_placeholder.pack()

    button_1 = tk.Button(
        master = frame_b, 
        command = lambda: [on_click(1), change_pic()],
        text = "1",
        width=25,
        height=2,
        foreground="white",
        background="black",
        font = 10)
    button_1.grid(row = 0, column=0)

    button_2 = tk.Button(
        master = frame_b, 
        command = lambda: [on_click(2), change_pic()],
        text = "2",
        width=25,
        height=2,
        foreground="white",
        background="black",
        font = 10)
    button_2.grid(row = 0, column=1)

    go_back_button = tk.Button(
        master = frame_c, 
        command = lambda: [go_back()],
        text = "Go back",
        width=25,
        height=1,
        foreground="white",
        background="black",
        font = 10)
    go_back_button.grid(row = 2, column=0)
    
    frame_d.pack()
    frame_b.pack()
    frame_a.pack()
    frame_c.pack()
    
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
#    QUINTARY     #
###################
def quintary():
    
    global image_index
    image_index = 0
    sequence = []

    #the sample of paths for images
    sample = random.sample(os.listdir("../../faces/high_quality_dataset/hot"), 10)
    sample = ["../../faces/high_quality_dataset/hot/" + choice for choice in sample]

    #it will be useful to keep the names of the pictures that the user is rating. Additionally, it might prove useful for future comparisons 
    #between users
    image_names = [re.search(r"image_\d+.jpg", path).group() for path in sample]

    images = [Image.open(x) for x in sample]
    images = [x.resize((400,500)) for x in images]

    def on_click(value):
        sequence.append(value)

    def change_pic(): 
        global image_index
        image_index += 1
        if image_index == len(images):
            window.destroy()
        else:
            photo = ImageTk.PhotoImage(images[image_index])
            picture_placeholder.configure(image=photo)
            picture_placeholder.image = photo 

            image_filename = image_names[image_index]
            filename_placeholder.configure(text = image_filename)
            filename_placeholder.text = image_filename

    def go_back():
        if len(sequence) == 0:
                    return
        global image_index
        image_index -= 1
        sequence.pop()

        photo = ImageTk.PhotoImage(images[image_index])
        picture_placeholder.configure(image=photo)
        picture_placeholder.image = photo 

        image_filename = image_names[image_index]
        filename_placeholder.configure(text = image_filename)
        filename_placeholder.text = image_filename

    window = tk.Tk()
    frame_image = ImageTk.PhotoImage(images[image_index])

    frame_a = tk.Frame()
    frame_b = tk.Frame()
    frame_c = tk.Frame()
    frame_d = tk.Frame()

    picture_placeholder = tk.Label(
        master = frame_a,
        image = frame_image,
        text = "Picture",
        foreground= "white",
        background="black"
        )
    picture_placeholder.pack()

    filename_placeholder = tk.Label(
        master = frame_d,
        text = image_names[image_index],
        foreground= "black",
        background= "white",
        height= 5,
        width = 50,
        font = 10
    )
    filename_placeholder.pack()

    button_1 = tk.Button(
        master = frame_b, 
        command = lambda: [on_click(1), change_pic()],
        text = "1",
        width=10,
        height=2,
        foreground="white",
        background="black",
        font = 10)
    button_1.grid(row = 0, column=0)

    button_2 = tk.Button(
        master = frame_b, 
        command = lambda: [on_click(2), change_pic()],
        text = "2",
        width=10,
        height=2,
        foreground="white",
        background="black",
        font = 10)
    button_2.grid(row = 0, column=1)

    button_3 = tk.Button(
        master = frame_b, 
        command = lambda: [on_click(3), change_pic()],
        text = "3",
        width=10,
        height=2,
        foreground="white",
        background="black",
        font = 10)
    button_3.grid(row = 0, column=2)

    button_4 = tk.Button(
        master = frame_b, 
        command = lambda: [on_click(4), change_pic()],
        text = "4",
        width=10,
        height=2,
        foreground="white",
        background="black",
        font = 10)
    button_4.grid(row = 0, column=3)

    button_5 = tk.Button(
        master = frame_b, 
        command = lambda: [on_click(5), change_pic()],
        text = "5",
        width=10,
        height=2,
        foreground="white",
        background="black",
        font = 10)
    button_5.grid(row = 0, column=4)

    go_back_button = tk.Button(
        master = frame_c, 
        command = lambda: [go_back()],
        text = "Go back",
        width=25,
        height=1,
        foreground="white",
        background="black",
        font = 10)
    go_back_button.grid(row = 2, column=0)
    
    frame_d.pack()
    frame_b.pack()
    frame_a.pack()
    frame_c.pack()
    
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
def rational():
    global image_index
    image_index = 0

    sequence = []

    sample = random.sample(os.listdir("../../faces/high_quality_dataset/hot"), 50)
    sample = ["../../faces/high_quality_dataset/hot/" + choice for choice in sample]

    image_names = [re.search(r"image_\d+.jpg", path).group() for path in sample]

    images = [Image.open(x) for x in sample]
    images = [x.resize((400,500)) for x in images]

    def change_picture_and_store_rating(event): 
        global image_index
        image_index += 1
        if 1 <= eval(entry.get()) <= 5:
            res.configure(text = "Result: " + str(eval(entry.get())))
            sequence.append(eval(entry.get()))
            if image_index == len(images):
                window.destroy()
            else:
                photo = ImageTk.PhotoImage(images[image_index])
                picture_placeholder.configure(image=photo)
                picture_placeholder.image = photo 

                image_filename = image_names[image_index]
                filename_placeholder.configure(text = image_filename)
                filename_placeholder.text = image_filename

    def go_back():
        if len(sequence) == 0:
                    return
        global image_index
        image_index -= 1
        sequence.pop()
        
        photo = ImageTk.PhotoImage(images[image_index])
        picture_placeholder.configure(image=photo)
        picture_placeholder.image = photo 

        image_filename = image_names[image_index]
        filename_placeholder.configure(text = image_filename)
        filename_placeholder.text = image_filename
              
                
    window = tk.Tk()
    frame_a = tk.Frame()
    frame_b = tk.Frame()
    frame_c = tk.Frame()
    frame_image = ImageTk.PhotoImage(images[image_index])


    picture_placeholder = tk.Label(
        master = frame_a,
        image = frame_image,
        text = "Picture",
        foreground= "white",
        background="black"
        )
    picture_placeholder.pack()

    filename_placeholder = tk.Label(
        master = frame_c,
        text = image_names[image_index],
        foreground= "white",
        background= "black",
        height= 2,
        width = 50,
        bd = 10,
        font= 10
    )
    filename_placeholder.pack()

    go_back_button = tk.Button(
        master = frame_b, 
        command = lambda: [go_back()],
        text = "Go back",
        width=25,
        height=1,
        foreground="black",
        background="white",
        bd = 0,
        font = 5,
        highlightbackground = "black", 
        highlightcolor="black",
        highlightthickness = 5)
    go_back_button.grid(row = 0, column=0)

    label = tk.Label(window, 
        text="You are rating in rational numbers from 1 to 5 (you can input digits after the comma)", 
        bd=40).pack()

    entry = tk.Entry(window, highlightthickness=5)
    entry.configure(highlightbackground="black", highlightcolor="black")
    entry.bind("<Return>", change_picture_and_store_rating)

    frame_c.pack()
    frame_a.pack()
    frame_b.pack()
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


if rating_choice == "binary":
    rated_dataset = binary()
elif rating_choice == "quintary":
    rated_dataset = quintary()
else:
    rated_dataset = rational()


filename = input("please enter your first and last name in this format: firstname_lastname  ")
filename = "datasets/" + filename + "_" + rating_choice + ".csv"

rated_dataset.to_csv(filename, encoding = 'utf-8', index = False)
