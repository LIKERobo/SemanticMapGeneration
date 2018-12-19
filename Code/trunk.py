#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 26 18:02:09 2018

@author: chen
"""

import numpy as np
import cv2
import math
import utils
from config import Config

class trunk(Config):
    """
    create the trunk part of the map, including the corridor, the pillar inside the doorway, and doors
    """
    def __init__(self,mapType):
        self.mapType = mapType
        self.gmap = np.ones(self.img_size,np.uint8)*self.colors['gray']
    
    def corridor(self):
        """
        create the corridor and set the positions of doors and rooms
        """
        # use a rectangle of a random size to emulate the corridor
        corridor_y_range = np.array([int(self.corridor_width[0]/self.resolution),int(self.corridor_width[1]/self.resolution)])
        corridor_x_range = np.array([int(self.corridor_length[0]/self.resolution),int(self.corridor_length[1]/self.resolution)])        
        corridor_size = [np.random.randint(corridor_y_range[0],corridor_y_range[1]),
                         np.random.randint(corridor_x_range[0],corridor_x_range[1])]       
        corridor_corners = [np.array(self.center)-corridor_size,np.array(self.center)+corridor_size]
        cv2.rectangle(self.gmap,tuple(corridor_corners[0]),tuple(corridor_corners[1]),self.colors['white'],-1)
        # find the contour of the corridor
        gradient= utils.calc_gradient(self.gmap)
        contour = np.where(gradient<0)
        contour = np.concatenate((contour[0][:,np.newaxis],contour[1][:,np.newaxis]),axis=1)
        # set the possible region of rooms and the number of rooms
        room_space = np.zeros((2,2*corridor_size[1]+1,2))
        num_room = np.random.randint(self.room_NumberSide[0],self.room_NumberSide[1],(2,))
        # choose the depth of doorway randomly
        self.groove_depth = np.random.randint(int(self.door_depth[0]/self.resolution),int(self.door_depth[1]/self.resolution),(2,))
        self.rooms_corners = []
        self.door_corners = []
        if self.mapType==0:
            self.circle_centers = []
            self.radius = []
        else:
            self.directions = []
        for side in range(2):
            # assign the space of each room
            room_space[side] = contour[contour[:,1] == corridor_corners[side][0]]
            rooms,rooms_corners_side = utils.room_division(room_space[side],num_room[side])        
            self.rooms_corners.append(rooms_corners_side)
            door_corners_side = []
            # assign the space of the door not against wall
            if self.mapType==0:  
                for room in rooms:
                    # set the size of door randomly
                    door_num=1
                    if room.shape[0]>280:
                        door_num = 2
                    door_corners_room = []
                    # small door
                    door_size = np.random.randint(int(self.singledoor_size[0]/self.resolution),int(self.singledoor_size[1]/self.resolution),size = door_num)
                    # large door with pillar
                    if door_num == 1:
                        if room.shape[0]>180:
                            if np.random.randint(6)==0:
                                door_size = np.random.randint(int(self.doubledoor_size[0]/self.resolution),int(self.doubledoor_size[1]/self.resolution),size = 1)
                    # set the position of small doorway
                    mittel = np.int(room.shape[0]/door_num)
                    for i in range(door_num):
                        single_size = door_size[i]
                        if mittel<=3*single_size:
                            door_center = room[int(mittel*(i+0.5)),:]
                        else:
                            door_range = room[mittel*i+int(1.5*single_size):mittel*(1+i)-int(1.5*single_size),:]
                            door_center_index  = np.random.randint(door_range.shape[0])
                            door_center = door_range[door_center_index,:]
                        door_corners_inside,door_corners_outside = utils.calc_corner(door_center,single_size,self.groove_depth,side)
                        # set the position of large doorway and pillar
                        if single_size>36:
                            r = np.random.choice(np.round(self.pillar_radius/self.resolution).astype(int))
                            door_corners_outside_up = door_corners_outside-np.array([[0,0],[(single_size/2).astype(np.int)+r+1,0]])
                            door_corners_outside_down = door_corners_outside+np.array([[(single_size/2).astype(np.int)+r+1,0],[0,0]])
                            door_corners_room.append(door_corners_outside_up)
                            door_corners_room.append(door_corners_outside_down)
                            center_pt =(np.sum(door_corners_inside+door_corners_outside,axis=0)/4).astype(np.int)
                            self.circle_centers.append(center_pt)
                            self.radius.append(r)
                        else:
                            door_corners_room.append(door_corners_outside)  
                        # draw the doorway on the map
                        cv2.rectangle(self.gmap,tuple(door_corners_inside[0][::-1]),tuple(door_corners_outside[1][::-1]),self.colors['white'],-1)
                    door_corners_side.append(door_corners_room)
            # assign the space of the door against wall
            else:
                directions_side = []
                for room in rooms:
                    # set the size of door randomly
                    door_num=1
                    door_size = np.random.randint(int(self.singledoor_size[0]/self.resolution),int(self.singledoor_size[1]/self.resolution),size = door_num)
                    # set the position of door near to the wall randomly
                    for i in range(door_num):
                        single_size = door_size[i]
                        if np.random.choice([0,1])==0:
                            door_range = room[math.ceil(single_size*11/14):math.floor(3*single_size/2),:]
                            direction = 0
                        else:
                            door_range = room[math.floor(-3*single_size/2):math.ceil(-single_size*11/14),:]
                            direction= 1
                        directions_side.append(direction)
                        door_center_index  = np.random.randint(door_range.shape[0])
                        door_center = door_range[door_center_index,:]
                        door_corners_inside,door_corners_outside = utils.calc_corner(door_center,single_size,self.groove_depth,side)
                        door_corners_side.append(door_corners_outside)
                        # draw the doorway on the map
                        cv2.rectangle(self.gmap,tuple(door_corners_inside[0][::-1]),tuple(door_corners_outside[1][::-1]),self.colors['white'],-1)
                self.directions.append(directions_side)
            self.door_corners.append(door_corners_side)

        return self.gmap, self.rooms_corners, self.door_corners,self.groove_depth
    
    def pillar(self,complete_map):
        """
        draw the pillar inside the large doorway
        """
        for circle_num, circle_center in enumerate(self.circle_centers):
            cv2.circle(complete_map,tuple(circle_center[::-1]),self.radius[circle_num],self.colors['gray'],-1) 
    
    def doors(self,complete_map,mask):
        """
        draw the doors on the map and label the doorway on the mask
        """
        for side in range(2):
            if self.mapType==0:
                # draw the doors not against the wall
                for i,door_room in enumerate(self.door_corners[side]):
                    for door in door_room:
                        # label the doorway on the mask
                        door_size = utils.draw_door(mask,complete_map,door,self.groove_depth,side)
                        # draw the door in a random orientation
                        door_start = door[0,:]
                        direction =1
                        switch = np.random.uniform(0,1)
                        if switch>=0.5:
                            door_start = door[1,:]
                            direction = -1
                        angle = np.random.uniform(math.pi/9,math.pi)
                        door_x = int(round(door_size*math.cos(angle)))
                        if door_x!=0:
                            door_x -=1
                        door_y = int(round(door_size*math.sin(angle)))
                        if door_y!=0:
                            door_y -=1            
                        door_end = np.array([door_start[0]+direction*door_x,door_start[1]-np.power(-1,side)*door_y])
                        cv2.line(complete_map,tuple(door_start[::-1]),tuple(door_end[::-1]),self.colors['black'],2)
            else:
                # draw the doors against the wall and the shadow under the door
                for i,door in enumerate(self.door_corners[side]):
                    door_size = utils.draw_door(mask,complete_map,door,self.groove_depth,side)
                    direction = self.directions[side][i]
                    door_start = door[direction,:]
                    door_end_y = np.round(np.sqrt(door_size**2-(self.rooms_corners[side][i][direction,0]-door_start[0])**2)).astype(np.uint8)
                    door_end = self.rooms_corners[side][i][direction,:]-np.array([0,np.power(-1,side)*door_end_y]) 
                    Polycorners = np.array([door_start[::-1],door_end[::-1]-np.array([0,np.power(-1,direction)]),self.rooms_corners[side][i][direction,:][::-1]]-np.array([0,np.power(-1,direction)]))
                    cv2.fillPoly(complete_map,[Polycorners],self.colors['gray'])
                    cv2.line(complete_map,tuple(door_start[::-1]),tuple(door_end[::-1]),self.colors['black'],2)

    
if __name__ == '__main__':
    trunk = trunk(0)
    gmap, rooms_corners, door_corners, door_depth= trunk.corridor()
    cv2.imshow('map',gmap)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
