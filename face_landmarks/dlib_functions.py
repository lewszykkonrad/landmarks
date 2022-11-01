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
import time
from shapely.geometry import Polygon

#A function that calculates an area based on given points
def Area(corners):
    n = len(corners) # of corners
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += corners[i][0] * corners[j][1]
        area -= corners[j][0] * corners[i][1]
    area = abs(area) / 2.0
    return area

#For dlib to work we need a detector and a predictor defined   
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("C:/Users/lewsz/OneDrive/Desktop/master_thesis/shape_predictor_68_face_landmarks_GTX.dat")


#This function is meant for processing the image by dlib
def facial_landmark_processor(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (400,500))
    
    rects = detector(image, 1)

    for (i, rect) in enumerate(rects):
        shape = predictor(image, rect)
        shape = face_utils.shape_to_np(shape)

    shape = shape.tolist()
    return shape


#This long function takes the object returned by facial_landmark_processor and performs calculations 
#on its landmakrs to obtain the needed parameters
def parameter_processor(shape):
    
    #creating a dictionary, that will be added later to a dataframe
    ratio_dictionary = {}
    
    #parameters that will be used for counting ratios and creating target variables
    
    right_eye_length = distance.euclidean(shape[42], shape[45])
    left_eye_length = distance.euclidean(shape[36], shape[39])
    
    right_eye_points = [shape[42], shape[43], shape[44], shape[45], shape[46], shape[47]]
    right_eye_points = [tuple(x) for x in right_eye_points]
    right_eye_area = Area(right_eye_points)
    
    left_eye_points = [shape[36], shape[37], shape[38], shape[39], shape[40], shape[41]]
    left_eye_points = [tuple(x) for x in left_eye_points]
    left_eye_area = Area(left_eye_points)
    
    
    eye_distance = distance.euclidean(shape[39], shape[42])
    eye_outer_distance = distance.euclidean(shape[36], shape[45])
    
    right_horizontal_eyebrow_width = distance.euclidean(shape[22], shape[26])
    left_horizontal_eyebrow_width = distance.euclidean(shape[17], shape[21])
    
    right_eyebrow_length = (distance.euclidean(shape[22], shape[23]) + 
                            distance.euclidean(shape[23], shape[24]) +
                            distance.euclidean(shape[24], shape[25]) +
                            distance.euclidean(shape[25], shape[26]))
                
    left_eyebrow_length = (distance.euclidean(shape[17], shape[18]) + 
                            distance.euclidean(shape[18], shape[19]) +
                            distance.euclidean(shape[19], shape[20]) +
                            distance.euclidean(shape[20], shape[21]))      
    
    right_vision_points = [shape[22], shape[23], shape[24], shape[25], shape[26], 
                           shape[45], shape[46], shape[47], shape[42]]
    right_vision_points = [tuple(x) for x in right_vision_points]
    right_vision_area = Area(right_vision_points)
    
    left_vision_points = [shape[17], shape[18], shape[19], shape[20], shape[21], 
                           shape[39], shape[40], shape[41], shape[36]]
    left_vision_points = [tuple(x) for x in left_vision_points]
    left_vision_area = Area(left_vision_points)
    
    
    
    nose_width = distance.euclidean(shape[31], shape[35])
    nose_length = distance.euclidean(shape[27], shape[33])
    
    nose_points = [shape[31], shape[32], shape[33], shape[34], shape[35], shape[27]]
    nose_points = [tuple(x) for x in nose_points]
    nose_area = Area(nose_points)
    
    mouth_length = distance.euclidean(shape[48], shape[54])
    mouth_width = distance.euclidean(shape[51], shape[57])
    
    upper_lip_points = [shape[48], shape[49], shape[50], shape[51], shape[52], shape[54],
                       shape[64], shape[63], shape[65], shape[66], shape[60]]
    upper_lip_points = [tuple(x) for x in upper_lip_points]
    upper_lip_area = Area(upper_lip_points)
    
    lower_lip_points = [shape[48], shape[59], shape[58], shape[57], shape[56], shape[55],
                       shape[54], shape[64], shape[65], shape[66], shape[67], shape[60]]
    lower_lip_points = [tuple(x) for x in lower_lip_points]
    lower_lip_area = Area(lower_lip_points)
    
    lip_area_points = [shape[48], shape[59], shape[58], shape[57], shape[56], shape[55],
                       shape[54], shape[53], shape[52], shape[51], shape[50], shape[49]]
    lip_area_points =    [tuple(x) for x in lip_area_points]
    lip_area = Area(lip_area_points)              
    
    left_cupid_lip_width = distance.euclidean(shape[50], shape[58])
    right_cupid_lip_width = distance.euclidean(shape[52], shape[56])
    
    chin_to_mouth_distance = distance.euclidean(shape[8], shape[57])
    mouth_to_nose_distance = distance.euclidean(shape[51], shape[33])
    nose_to_chin_distance = distance.euclidean(shape[8], shape[33])
    
    left_eye_corner_to_left_lip_corner_length = distance.euclidean(shape[36], shape[49])
    right_eye_corner_to_left_lip_corner_length = distance.euclidean(shape[45], shape[54])
    
    left_lip_corner_to_chin_distance = distance.euclidean(shape[48], shape[8])
    right_lip_corner_to_chin_distance = distance.euclidean(shape[54], shape[8])
    
    left_bottom_face_to_left_lip_corner_length = distance.euclidean(shape[4], shape[48])
    right_bottom_face_to_right_lip_corner_length = distance.euclidean(shape[12], shape[54])
    
    face_length = abs(shape[19][1] - shape[8][1])
    
    face_width_at_bottom = distance.euclidean(shape[4], shape[12])
    face_width_at_top = distance.euclidean(shape[1], shape[15])
    
    face_points = [shape[19], shape[18], shape[17],shape[0], shape[1], shape[2], shape[3], 
                   shape[4], shape[5], shape[6],shape[7], shape[8], shape[9], shape[0], 
                   shape[11], shape[12],shape[13], shape[14], shape[15], shape[16], 
                   shape[26], shape[25], shape[24]]
    face_points = [tuple(x) for x in face_points]
    face_area = Area(face_points)
    
    inter_nose_mouth_area_points = [shape[31], shape[32], shape[33], shape[34], shape[35], shape[54], shape[48]]
    inter_nose_mouth_area_points = [tuple(x) for x in inter_nose_mouth_area_points]
    inter_nose_mouth_area = Area(face_points)
    
    under_mouth_points = [shape[48], shape[54], shape[10], shape[9], shape[8], shape[7], shape[6]]
    under_mouth_points = [tuple(x) for x in under_mouth_points]
    under_mouth_area = Area(under_mouth_points)
    
    bottom_face_points = [shape[5], shape[6], shape[7], shape[8], shape[9], shape[10], shape[11]]
    bottom_face_points = [tuple(x) for x in bottom_face_points]
    bottom_face_area = Area(bottom_face_points)
    
    lower_face_points = [shape[5], shape[4], shape[33], shape[13], shape[12], shape[11]]
    lower_face_points = [tuple(x) for x in lower_face_points]
    lower_face_area = Area(lower_face_points)
    
    upper_face_points = [shape[13], shape[14], shape[15], shape[1], shape[2], shape[3]]
    upper_face_points = [tuple(x) for x in upper_face_points]
    upper_face_area = Area(upper_face_points)
    
    top_face_points = [shape[1], shape[0], shape[17], shape[18], shape[19], shape[24],
                       shape[25], shape[26], shape[16], shape[15], shape[60]]
    top_face_points = [tuple(x) for x in top_face_points]
    top_face_area = Area(top_face_points)
    
    inner_triangle_points = [shape[0], shape[16], shape[8]]
    inner_triangle_points = [tuple(x) for x in inner_triangle_points]
    inner_triangle_area = Area(inner_triangle_points)
    
    right_outer_face_points = [shape[8], shape[9], shape[10], shape[11], shape[12], shape[13],
                       shape[14], shape[15], shape[16]]
    right_outer_face_points = [tuple(x) for x in right_outer_face_points]
    right_outer_face_area = Area(right_outer_face_points)
    
    left_outer_face_points = [shape[0], shape[1], shape[2], shape[3], shape[4], shape[5],
                       shape[6], shape[7], shape[8]]
    left_outer_face_points = [tuple(x) for x in left_outer_face_points]
    left_outer_face_area = Area(left_outer_face_points)
    
    features_points = [shape[17], shape[18], shape[19], shape[24], shape[25], shape[26],
                    shape[54], shape[55], shape[56], shape[57], shape[58], shape[59], shape[48]]
    features_points = [tuple(x) for x in features_points]
    features_area = Area(features_points)
    
    
    # CALCULATING RATIOS
    
    #symmetry
    eye_area_symmetry = (right_eye_area / left_eye_area)
    ratio_dictionary['s1'] = eye_area_symmetry
    
    eye_length_symmetry = (right_eye_length/left_eye_length)
    ratio_dictionary['s2'] = eye_length_symmetry


    
    #area ratios
    eyes_to_lips = ((right_eye_area + left_eye_area) / (lip_area))
    ratio_dictionary['a1'] = eyes_to_lips
    
    eyes_to_nose = ((right_eye_area + left_eye_area) / nose_area)
    ratio_dictionary['a2'] = eyes_to_nose
    
    eyes_to_face = ((right_eye_area + left_eye_area) / face_area)
    ratio_dictionary['a3'] = eyes_to_face
    
    eyes_to_top_face = ((right_eye_area + left_eye_area) / top_face_area)
    ratio_dictionary['a4'] = eyes_to_top_face
    
    lips_to_nose = ((lip_area) / nose_area)
    ratio_dictionary['a5'] = lips_to_nose
    
    lips_to_face = ((lip_area) / face_area)
    ratio_dictionary['a6'] = lips_to_face
    
    lips_to_lower_face = ((lip_area) / lower_face_area)
    ratio_dictionary['a7'] = lips_to_lower_face
    
    nose_to_face = (nose_area / face_area)
    ratio_dictionary['a8'] = nose_to_face
    
    eyes_lips_nose_to_face = ((right_eye_area + left_eye_area + lip_area + nose_area) / face_area)
    ratio_dictionary['a9'] = eyes_lips_nose_to_face
    
    eyes_lips_nose_to_inner_face = ((right_eye_area + left_eye_area + lip_area + nose_area) 
                                    / inner_triangle_area)
    ratio_dictionary['a10'] = eyes_lips_nose_to_inner_face
    
    features_to_triangle = (features_area / inner_triangle_area)
    ratio_dictionary['a11'] = features_to_triangle
    
    features_to_face = features_area / face_area
    ratio_dictionary['a12'] = features_to_face
    
    features_to_outer = features_area / (left_outer_face_area + right_outer_face_area)
    ratio_dictionary['a13'] = features_to_outer
    
    inner_to_outer = (inner_triangle_area / (left_outer_face_area + right_outer_face_area))
    ratio_dictionary['a14'] = inner_to_outer
    
    inner_to_face = inner_triangle_area / face_area
    ratio_dictionary['a15'] = inner_to_face
    
    outer_to_face = (left_outer_face_area + right_outer_face_area) / face_area
    ratio_dictionary['a16'] = outer_to_face
    
    top_face_to_face = top_face_area / face_area
    ratio_dictionary['a17'] = top_face_to_face
    
    upper_face_to_face = upper_face_area / face_area
    ratio_dictionary['a18'] = upper_face_to_face
    
    lower_face_to_face = lower_face_area / face_area
    ratio_dictionary['a19'] = lower_face_to_face
    
    bottom_face_to_face = bottom_face_area / face_area
    ratio_dictionary['a20'] = bottom_face_to_face
    
    upper_to_bottom = upper_face_area / bottom_face_area
    ratio_dictionary['a21'] = upper_to_bottom
    
    vision_to_face = (right_vision_area + left_vision_area) / face_area
    ratio_dictionary['a22'] = vision_to_face

    nose_lips_eyes_to_features_area = (nose_area + right_eye_area + left_eye_area + lip_area) / features_area
    ratio_dictionary['a23'] = nose_lips_eyes_to_features_area
    
    #distances
    
    face_length_to_width_top = (face_length / face_width_at_top)
    ratio_dictionary['d1'] = face_length_to_width_top
    
    face_length_to_width_bottom = (face_length / face_width_at_bottom)
    ratio_dictionary['d2'] = face_length_to_width_bottom
    
    face_top_to_bottom_width = face_width_at_top/face_width_at_bottom
    ratio_dictionary['d3'] = face_top_to_bottom_width
    
    eye_distance_to_face_width = (eye_distance /  face_width_at_top)
    ratio_dictionary['d4'] = eye_distance_to_face_width
    
    eyebrows_to_face_width = ((right_eyebrow_length + left_eyebrow_length) / face_width_at_top)
    ratio_dictionary['d5'] = eyebrows_to_face_width
    
    mouth_to_eye_distance = (mouth_length / eye_distance)
    ratio_dictionary['d6'] = mouth_to_eye_distance
    
    mouth_to_eye_spread = (mouth_length / eye_outer_distance)
    ratio_dictionary['d7'] = mouth_to_eye_spread
    
    mouth_to_nose_width = (mouth_length / nose_width)
    ratio_dictionary['d8'] = mouth_to_nose_width
    
    nose_length_to_face_length = (nose_length/face_length)
    ratio_dictionary['d9'] = nose_length_to_face_length
    
    bottom_distance_to_face = (chin_to_mouth_distance / face_length)
    ratio_dictionary['d10'] = bottom_distance_to_face
    
    nose_to_mouth_to_face =   (mouth_to_nose_distance / face_length)
    ratio_dictionary['d11'] = nose_to_mouth_to_face
       
    eye_distance_to_nose_width = (eye_distance / nose_width)
    ratio_dictionary['d12'] = eye_distance_to_nose_width
    
    nose_length_to_nose_to_chin_distance = nose_length / nose_to_chin_distance
    ratio_dictionary['d13'] = nose_length_to_nose_to_chin_distance
    
    nose_width_to_nose_mouth_distance = nose_width / mouth_to_nose_distance
    ratio_dictionary['d14'] = nose_width_to_nose_mouth_distance

    mouth_length_to_width = mouth_length / mouth_width
    ratio_dictionary['d15'] = mouth_length_to_width

    features_length_to_face_width = ((right_eyebrow_length + left_eyebrow_length + mouth_length + left_eyebrow_length + right_eyebrow_length) / face_width_at_top)
    ratio_dictionary['d16'] = features_length_to_face_width

    features_length_to_face_length = ((right_eyebrow_length + left_eyebrow_length + mouth_length + left_eyebrow_length + right_eyebrow_length) / face_length)
    ratio_dictionary['d17'] = features_length_to_face_length
    
    
    
    #distances and areas
    
    eyebrows_to_eyes = ((right_eyebrow_length + left_eyebrow_length) / (right_eye_area + left_eye_area))
    ratio_dictionary['da1'] = eyebrows_to_eyes
    
    vision_to_face_width = ((right_vision_area + left_vision_area) / face_width_at_top)
    ratio_dictionary['da2'] = vision_to_face_width

    features_length_to_features_area = ((right_eyebrow_length + left_eyebrow_length + mouth_length + left_eyebrow_length + right_eyebrow_length) / features_area)
    ratio_dictionary['da3'] = features_length_to_features_area
    
                                                 
    return ratio_dictionary