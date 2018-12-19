#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base Configurations class.

Created on Wed Dec 12 13:44:49 2018

@author: chen
"""
import numpy as np

class Config():
    """
    the tunable parameters
    """
    colors ={'gray':127,'white':254,'black':0}
    # the physical size associated with one pixel
    resolution = 0.05
    # the size of the whole simulated map
    img_size = (512,512)
    center  = [int(img_size[0]/2),int(img_size[1]/2)]
    # the optional range of corridor's size
    corridor_width = [1.2,1.8]
    corridor_length = [8.5,10]
    # the optional range of doorway's size
    singledoor_size = [0.7,1.7]
    doubledoor_size = [2.8,3.2]
    door_depth = [0.1,0.45]
    # the optional range of number of rooms on one side
    room_NumberSide = [1,3]
    # the optional range of width of room
    room_width = [4.2,7.8]
    # the optional range of radius of pillar in the middle of doorway
    pillar_radius = np.array([0.2,0.25,0.3])
    # the remained space around the doorway for dataExtraction
    x_gap = 0.6
    y_gap = 0.4
    # the upper limit of the number of generated data
    cnn_num = 1000
    unet_num = 1000
    # the stride of sliding window for dataExtraction
    stride =8
    # the size of sliding window
    window_size =64
    # the threshold of classified as a door
    thres_door=1
    # the threshold of classified as background
    thres_bg = 0.3
    # the save path of genereated data
    dir_path = ['cnn_train','unet_train']
    def __init__(self):
        pass