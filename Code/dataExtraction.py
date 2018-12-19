#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 21:38:07 2018

@author: chen
"""

import numpy as np
import cv2
import utils
import orig_map
import addNoise
import sys
from config import Config

class dataExtraction(Config):
    """
    extract and save the data from the simulated maps automatically
    """
    def __init__(self):
        self.cnn_count = 0
        self.unet_count = 0
    def data(self,noised_maps,ref_map,mask,rotate_angle,mode):
        # Lboxes are the large boxes around doorways for classfication
        # Sboxes are the small boxes around doorways for determination of the number of doors involved in a window
        # divided_masks are the masks of each doorway separately
        Lboxes=[]
        Sboxes=[]
        divided_masks=[]
        _,contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            # separate the masks based on their contours
            canvas = np.zeros_like(mask)
            cv2.drawContours(canvas,[cnt], 0, 255, -1)
            divided_masks.append(canvas)
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)  
            box = np.int0(np.round(box))
            # adjust the large box according to the rotated angle
            tuned_x_gap = round(self.x_gap/self.resolution*(1-utils.normfun(rotate_angle%90,24)*33)).astype(np.int)
            large_box = utils.calc_box(box,tuned_x_gap,int(self.y_gap/self.resolution),rotate_angle,self.center) 
            Lboxes.append(large_box)
            small_box = utils.calc_box(box,1,1,rotate_angle,self.center)
            Sboxes.append(small_box)
        Lboxes = np.array(Lboxes)
        Sboxes = np.array(Sboxes) 
        # rotate the divided masks with the given angle
        rotate_kernel = cv2.getRotationMatrix2D(tuple(self.center),rotate_angle,1)
        rotate_masks=[]
        for mask in divided_masks:
            rotate_mask = cv2.warpAffine(mask,rotate_kernel,self.img_size,borderValue=0)
            rotate_masks.append(rotate_mask)
          
        for x in range(0,self.img_size[0],self.stride):
            if (x+self.window_size)<=self.img_size[0]:
                for y in range(0,self.img_size[1],self.stride):
                    if (y+self.window_size)<=self.img_size[1]:
                        window = np.array([x,y,x+self.window_size-1,y+self.window_size-1])
                        # reious is regarded as the threshold for classification
                        reious = utils.overlapp(window,Lboxes)
                        if mode ==0:
                            # extract samples of background
                            if (reious<=self.thres_bg).all(): 
                                bk = ref_map[x:x+self.window_size,y:y+self.window_size]
                                if np.sum(bk==self.colors['gray'])<3600:
                                    # calculate the selection rate according to the variance and the proportion of gray pixels
                                    var = np.var(bk)
                                    denominator = 40-utils.sigmoid(var/3200)*12+30*np.sum((bk>120)*(bk<130))/(self.window_size**2)
                                    if np.random.randint(denominator)==0:  
                                        # save the extracted samples
                                        for n,nmaps in enumerate(noised_maps):
                                            for l, lmap in enumerate(nmaps):
                                                background = lmap[x:x+self.window_size,y:y+self.window_size]
                                                # background region for cnn
                                                path = self.dir_path[0]+'_'+str(n)+str(l)+'/0/'
                                                utils.mkdir(path)
                                                cv2.imwrite(path+str(self.cnn_count)+'.png',background)
                                        self.cnn_count +=1
                                        # set the upper limit
                                        if self.cnn_count==self.cnn_num:
                                            sys.exit()
                        if mode == 1:
                            # extract samples of doorways
                            if(reious>=self.thres_door).any(): 
                                # calculate the number of doors involved in this window
                                if (rotate_angle%90)<10:
                                    thres =0
                                elif (rotate_angle%90)>80:
                                    thres =0
                                elif (rotate_angle%90)<=45:
                                    thres = (rotate_angle%90)/45*0.5
                                else :
                                    thres = (90-rotate_angle%90)/45*0.5
                                num_door = np.where(utils.overlapp(window,Sboxes)>thres)[0].shape[0]
                                ref = np.where(reious>=self.thres_door)[0].shape[0]
                                # ensure that all the doors involved in this window are completely covered
                                if num_door ==ref:     
                                    if np.random.randint(10)==0:
                                        indices = np.where(reious>=self.thres_door)[0]
                                        # save the extracted samples
                                        for n,nmaps in enumerate(noised_maps):
                                            for l, lmap in enumerate(nmaps):
                                                proxi_area = lmap[x:x+self.window_size,y:y+self.window_size]
                                                mask_area = np.zeros((self.window_size,self.window_size),np.uint8)
                                                for index in indices:
                                                    mask_area = mask_area+rotate_masks[index][x:x+self.window_size,y:y+self.window_size]
                                                # doorway region for u-net
                                                path = self.dir_path[1]+'_'+str(n)+str(l)+'/imgs/'
                                                utils.mkdir(path)
                                                cv2.imwrite(path+str(self.unet_count)+'.png',proxi_area)
                                                # mask of doorway for u-net
                                                path = self.dir_path[1]+'_'+str(n)+str(l)+'/masks/'
                                                utils.mkdir(path)
                                                cv2.imwrite(path+str(self.unet_count)+'.png',mask_area)
                                                # doorway region for cnn
                                                path = self.dir_path[0]+'_'+str(n)+str(l)+'/1/'
                                                utils.mkdir(path)
                                                if self.cnn_count<self.cnn_num:
                                                    cv2.imwrite(path+str(self.cnn_count)+'.png',proxi_area)
                                        self.cnn_count+=1
                                        self.unet_count+=1 
                                        # set the upper limit
                                        if self.unet_count==self.unet_num:
                                            sys.exit()
                                            
if __name__=='__main__': 
    map_num=5
    noise_types = ['combindNoise','GaussNoise']
    levels=[1,7]   
    map_sim = orig_map.orig_map()
    maps,masks = map_sim.creater(map_num)
    extract = dataExtraction()
    for single_map,single_mask in zip(maps,masks):
        noised_maps = []
        rotate_angle = np.random.randint(0,360)
        addnoise = addNoise.addNoise(single_map)
        ref_map = addnoise.add_noise('noNoise',rotate_angle,0)
        for noise in noise_types:
            multilevels_maps = []
            for l in levels:    
                noise_map= addnoise.add_noise(noise,rotate_angle,l)
                multilevels_maps.append(noise_map)
            noised_maps.append(multilevels_maps)
        extract.data(noised_maps,ref_map,single_mask,rotate_angle,1)
    
        