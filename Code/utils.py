#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Common utility functions 

Created on Sun May 27 16:37:42 2018

@author: chen
"""
import math
import cv2
import os
from imutils import paths
import numpy as np
import scipy.ndimage

def rotate_cooridinate(cooridinate_og,rotate_angle,rotate_center):
    """
    calculate the coordinates after rotation
    """
    rotate_angle = rotate_angle*(math.pi/180)
    rotated_x = (cooridinate_og[0]-rotate_center[0])*math.cos(rotate_angle)\
                -(cooridinate_og[1]-rotate_center[1])*math.sin(rotate_angle)+rotate_center[0]
    rotated_y = (cooridinate_og[0]-rotate_center[0])*math.sin(rotate_angle)\
                +(cooridinate_og[1]-rotate_center[1])*math.cos(rotate_angle)+rotate_center[1]
    rotated_coordinate = np.array([rotated_x,rotated_y])
    rotated_coordinate = np.round(rotated_coordinate).astype(np.int)
    return rotated_coordinate

def mkdir(path):
    """
    create new folder automatically
    """
    folder = os.path.exists(path)

    if not folder:                  
        os.makedirs(path)
        
def load_data(path):
    """
    load data from specified folder
    """
    print("[INFO] loading images...")
    imgs = []    
    # grab the image paths and randomly shuffle them
    imagePaths = sorted(list(paths.list_images(path)))

    for imagePath in imagePaths:
        # load the image, pre-process it, and store it in the data list
        image = cv2.imread(imagePath,cv2.IMREAD_GRAYSCALE)
        imgs.append(image)
    return imgs

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def normfun(x,sigma):
    """
    function of normal distribution
    """
    mu = 45    
    pdf = np.exp(-((x - mu)**2)/(2*sigma**2)) / (sigma * np.sqrt(2*np.pi))
    return pdf

def calc_box(box,x_gap,y_gap,rotate_angle,center):
    """
    calculate the size of the required surrounding environment for doorway segmentation
    box: four corners' coordinates of doorway
    x_gap: remained space in the vertical way
    y_gap: remained space in the horizontal way
    """
    door_box = np.array([box[0][::-1]+[y_gap,x_gap],box[1][::-1]+[y_gap,-x_gap],
                         box[2][::-1]-[y_gap,x_gap],box[3][::-1]-[y_gap,-x_gap]])
    rotated_box = []
    for coordinate in door_box:
        box_coordinate = rotate_cooridinate(coordinate,rotate_angle,center)
        rotated_box.append(box_coordinate)

    rotated_box = np.array(rotated_box)
    box = [np.min(rotated_box[:,0]),np.min(rotated_box[:,1]),np.max(rotated_box[:,0]),np.max(rotated_box[:,1])]
    return box

def calc_IoU(candidateBound, groundTruthBounds):
    """
    calculate the intersection over union
    """
    cx1 = candidateBound[0]
    cy1 = candidateBound[1]
    cx2 = candidateBound[2]
    cy2 = candidateBound[3]

    gx1 = groundTruthBounds[:,0]
    gy1 = groundTruthBounds[:,1]
    gx2 = groundTruthBounds[:,2]
    gy2 = groundTruthBounds[:,3]

    carea = (cx2 - cx1) * (cy2 - cy1) 
    garea = (gx2 - gx1) * (gy2 - gy1) 

    x1 = np.maximum(cx1, gx1)
    y1 = np.maximum(cy1, gy1)
    x2 = np.minimum(cx2, gx2)
    y2 = np.minimum(cy2, gy2)
    w = np.maximum(0, x2 - x1)
    h = np.maximum(0, y2 - y1)
    area = w * h 

    ious = area / (carea + garea - area)

    return ious

def overlapp(candidateBound, groundTruthBounds):
    """
    calculate the proportion of prediction to groundtruth
    """
    cx1 = candidateBound[0]
    cy1 = candidateBound[1]
    cx2 = candidateBound[2]
    cy2 = candidateBound[3]

    gx1 = groundTruthBounds[:,0]
    gy1 = groundTruthBounds[:,1]
    gx2 = groundTruthBounds[:,2]
    gy2 = groundTruthBounds[:,3]

    garea = (gx2 - gx1) * (gy2 - gy1) 

    x1 = np.maximum(cx1, gx1)
    y1 = np.maximum(cy1, gy1)
    x2 = np.minimum(cx2, gx2)
    y2 = np.minimum(cy2, gy2)
    w = np.maximum(0, x2 - x1)
    h = np.maximum(0, y2 - y1)
    area = w * h 

    reious = area / garea

    return reious

def calc_corner(door_center,door_size,door_depth,side):
    """
    calculate the corners' coordinates from the centroid, size and depth of doorway
    door_corners_inside is a list of coordinates of corners close to the corridor
    door_corners_outside is a list of coordinates of corners close to the room
    """
    door_corners_inside = [door_center-np.array([np.int(door_size/2),0]),
                           door_center+np.array([door_size-np.int(door_size/2),0])]
    door_corners_outside = [x-np.array([0,np.power(-1,side)*door_depth[side]]) 
                            for x in door_corners_inside]
    door_corners_outside = np.array(door_corners_outside)

    return door_corners_inside,door_corners_outside

def draw_door(mask,complete_map,door,door_depth,side):
    """
    label the doorway on the mask and add some error inside the doorway region
    """
    door_size = abs(door[1,0]-door[0,0])
    door_area_inside = door+np.array([0,np.power(-1,side)*door_depth[side]])
    # label the doorway on the mask
    cv2.rectangle(mask,tuple(door[0][::-1]),tuple(door_area_inside[1][::-1]),255,-1)
    # add a small point to emulate the error in the doorway region
    if door_size>20:
        if np.random.randint(4)==0:
            if side ==0:
                pt_center = [np.random.randint(door[0,0]+4,door[1,0]-3),np.random.randint(door[0,1],door_area_inside[0,1])]
            else:
                pt_center = [np.random.randint(door[0,0]+3,door[1,0]-2),np.random.randint(door_area_inside[0,1],door[0,1])]
            cv2.circle(complete_map,tuple(pt_center[::-1]),np.random.choice([1,2,3]),0,-1)
    return door_size   

def room_division(room_space,num_room):
    """
    assign the lengths of rooms according to the length of corridor and number of rooms
    room_space: coordinates of corridor's side
    num_room: the number of rooms on one side
    rooms: a list of the coordinates belonging to different rooms
    rooms_corners: a list of only the top and bottom cooridnates of different rooms
    """
    rooms = []
    rooms_corners=[]
    a = num_room
    thickness = np.random.randint(2,5)
    length = room_space.shape[0]-(num_room-1)*thickness
    start_point = 0
    for i in range(num_room-1):
        room_size = np.random.randint(length/(a+0.7),length/(a-0.7))
        room = room_space[start_point:start_point+room_size,:]
        rooms.append(room)
        start_point +=room_size+thickness
    room = room_space[start_point:,:]
    rooms.append(room)
    rooms = [room.astype(np.int) for room in rooms]
    for x in rooms:
        rooms_corner = np.concatenate((x[0,:][np.newaxis,:],x[-1,:][np.newaxis,:]),axis = 0)
        rooms_corners.append(rooms_corner)
        
    return rooms,rooms_corners
        
def calc_gradient(gmap):
    """
    calculate the gradient of image to find the contour
    """
    kernel = np.array([[1,1,1],[1,-8,1],[1,1,1]])
    img = gmap.astype(np.int16)
    gradient = scipy.ndimage.correlate(img,kernel,mode = 'constant',cval =127)
    
    return gradient

        
    

    
         
        
        