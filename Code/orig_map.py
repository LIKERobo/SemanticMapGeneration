#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 14:58:58 2018

@author: chen
"""
import numpy as np
import cv2
import utils
import trunk
import addObjects
from config import Config

class orig_map(Config):
    """
    create the original map without noise and only in the vertical direction
    outputs are the simulated map and its corresponding mask
    """
    def __init__(self):
        pass
    
    def creater(self,map_num):
        maps=[]
        masks=[]
        for _ in range(map_num):
            complete_map = np.ones(self.img_size,np.uint8)*self.colors['gray']
            mask = np.zeros_like(complete_map,np.uint8)
            grooves_canvas = np.zeros_like(complete_map,np.uint8)
            # mapType 0: door aganist the wall, mapType 1: door not aganist the wall
            if np.random.randint(2)==0:
                mapType = 0
            else:
                mapType = 1
            map_trunk = trunk.trunk(mapType)
            # create the corridor and set the positions of doors and rooms
            corridor_map, rooms_corners, door_corners,doorway_depth= map_trunk.corridor()
            complete_map[corridor_map==self.colors['white']]=self.colors['white']
            # process each room
            for side in range(2):            
                for i,room in enumerate(rooms_corners[side]):
                    # create one room
                    room_map = np.zeros_like(complete_map,np.uint8)
                    room_width = np.random.randint(int(self.room_width[0]/self.resolution),int(self.room_width[1]/self.resolution))
                    room_length = abs(room[1,0]-room[0,0])
                    room_corner_inside = room[0,:]-np.power(-1,side)*np.array([0,doorway_depth[side]])
                    room_corner_outside = room[1,:]-np.power(-1,side)*np.array([0,room_width])
                    cv2.rectangle(room_map,tuple(room_corner_inside[::-1]),tuple(room_corner_outside[::-1]),self.colors['white'],-1)
                    # add grooves and furnitures to this room
                    door_corner = door_corners[side][i]
                    add = addObjects.addObjects(room_map,room_width,room_length)
                    add.grooves(door_corner,grooves_canvas,mapType)
                    add.furnitures()                        
                    complete_map[room_map==self.colors['white']]=self.colors['white']
            # add the pillar inside the large door                           
            if mapType==0:
                map_trunk.pillar(complete_map)
            # draw the contour
            gradient = utils.calc_gradient(complete_map)
            complete_map[gradient!=0]=self.colors['black'] 
            # dilate the grooves to break the contour
            kernel = np.ones((np.random.choice([1,2,3]),np.random.choice([1,2,3])),np.uint8)               
            grooves_canvas = cv2.dilate(grooves_canvas,kernel,iterations = 1)
            complete_map[grooves_canvas==self.colors['white']]=self.colors['gray']
            #draw the doors on the map and label the doorway on the mask
            map_trunk.doors(complete_map,mask)
                
            maps.append(complete_map)
            masks.append(mask)
      
        return maps,masks

if __name__=='__main__':
    map_sim = orig_map()
    map_num = 1
    maps,masks = map_sim.creater(map_num)
    maps = np.array(maps)
    masks = np.array(masks)
#    np.save('maps_5000_0',maps)
#    np.save('masks_5000_0',masks)
    for single_map,single_mask in zip(maps,masks):
        cv2.imshow('map',single_map)
        cv2.imshow('mask',single_mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
