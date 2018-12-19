#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 14:35:25 2018

@author: chen
"""
import numpy as np
import cv2
import utils
from config import Config

class addObjects(Config):
    """
    add objects to the room
    room_map is the map of each room
    """
    def __init__(self,room_map,room_width,room_length):
        self.room_map = room_map
        self.room_width = room_width
        self.room_length = room_length
        self.room_area = self.room_width*self.room_length
        
    def grooves(self,door_corner,grooves_canvas,mapType):
        """
        add grooves ot emulate the measurement errors on the contours
        """
        # the number of grooves is related to the area of the room
        num_groove = np.random.randint(self.room_area*self.resolution**2/5,self.room_area*self.resolution**2/2)
        for num in range(num_groove):
            # find the contour of the room
            gradient= utils.calc_gradient(self.room_map)
            contour = np.where(gradient<0)
            contour = np.concatenate((contour[0][:,np.newaxis],contour[1][:,np.newaxis]),axis=1)
            # choose the position of the groove on the contour randomly
            index = np.random.randint(contour.shape[0])
            groove_ref = contour[index]
            # set the range of groove's size
            groove_size = np.minimum(self.room_width,self.room_length)/24
            # choose the coordinates of cornes randomly based on groove_ref and groove_size
            groove_corner_a = groove_ref - np.random.randint(0,groove_size,2)
            groove_corner_b = groove_ref + np.array([np.random.randint(0,groove_size),-np.random.randint(0,groove_size)])
            groove_corner_c = groove_ref + np.random.randint(0,groove_size,2)   
            groove_corner_d = groove_ref - np.array([np.random.randint(0,groove_size),-np.random.randint(0,groove_size)])
            groove_corners = [groove_corner_a[::-1],groove_corner_b[::-1],groove_corner_c[::-1],groove_corner_d[::-1]]
            groove_corners = np.asarray(groove_corners)
            # determine the region of doorway
            if mapType==0:   
                door_space_0 = np.arange(door_corner[0][0,0],door_corner[0][-1,0]+1).reshape(-1,1)
                if len(door_corner)>1:    
                    for j in range(1,len(door_corner)):
                        single_door_space_0 = np.arange(door_corner[j][0,0],door_corner[j][-1,0]+1).reshape(-1,1)
                        door_space_0 = np.concatenate((door_space_0,single_door_space_0),axis=0)
                door_space_1 = np.ones_like(door_space_0,dtype=np.int)*door_corner[0][0,1]
            else:
                door_space_0 = np.arange(door_corner[0,0],door_corner[-1,0]+1).reshape(-1,1)           
                door_space_1 = np.ones_like(door_space_0,dtype=np.int)*door_corner[0,1]
            door_space = np.concatenate((door_space_0,door_space_1),axis=1)
            # ensure that the grooves are not on the doorway region
            cv2.fillPoly(self.room_map,[groove_corners],self.colors['gray'])
            for k in range(door_space.shape[0]):
                if self.room_map[door_space[k,0],door_space[k,1]]==self.colors['gray']:
                    cv2.fillPoly(self.room_map,[groove_corners],self.colors['white'])
            # save the grooves in the variable grooves_canvas
            grooves_canvas[self.room_map==self.colors['gray']] = self.colors['white']
    
    def furnitures(self):
        """
        add the furnitures into the room
        """
        for _ in range(int(self.room_area*self.resolution**2/6)):
            # choose the position of furniture in the room randomly
            pt =[np.random.choice(np.where(self.room_map==self.colors['white'])[m]) for m in range(2)]
            # calculate the distance between the furniture and wall
            _,binary = cv2.threshold(self.room_map,200,255,cv2.THRESH_BINARY)
            _,cnt,_  = cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            main_cnt_index = np.argmax([single_cnt.shape[0] for single_cnt in cnt])
            dist = cv2.pointPolygonTest(cnt[main_cnt_index],tuple(pt[::-1]),measureDist=True)
            if dist>0.25/self.resolution:
                # use circle to emulate the furniture
                if np.random.uniform(0,1)>=0.5:
                    max_size = np.minimum(int(dist-0.1/self.resolution),int(0.6/self.resolution))
                    radius = np.random.randint(1,max_size)
                    cv2.circle(self.room_map,tuple(pt[::-1]),radius,127,-1)
                # use rectangle to emulate the furniture
                else:
                    max_size = np.minimum(int(dist-0.1/self.resolution),int(0.6/self.resolution))
                    width = np.random.randint(1,max_size)
                    heigh = np.random.randint(1,max_size)
                    object_angle = np.random.randint(0,360)
                    object_corners_og = [[pt[1]+width,pt[0]+heigh],[pt[1]+width,pt[0]-heigh],
                                         [pt[1]-width,pt[0]-heigh],[pt[1]-width,pt[0]+heigh]]
                    # rotate the rectangles
                    object_corners = []
                    for point in object_corners_og:
                        point_new = utils.rotate_cooridinate(point,object_angle,pt[::-1])
                        object_corners.append(point_new)
                    object_corners = np.asarray(object_corners)
                    cv2.fillPoly(self.room_map,[object_corners],self.colors['gray'])
