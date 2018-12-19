#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 22:14:36 2018

@author: chen
"""
import numpy as np
import cv2
import skimage
import orig_map
from config import Config

class addNoise(Config):
    """
    add different types of noise on differet levels to the map
    """
    def __init__(self,single_map):
        self.map = single_map

    def multiNoise(self,rotate_angle,level):
        # add salt noise to the contours of rooms and furnitures
        sp_map = skimage.util.random_noise(self.map,'salt',amount=np.random.uniform(0.05+level*0.1,0.15+level*0.1))
        sp_map = np.round(sp_map*255).astype(np.uint8)
        sp_map[self.map==self.colors['gray']]=self.colors['gray']
        sp_map[self.map==self.colors['white']]=self.colors['white']
        # find the region surrounding the contours
        blur_map = cv2.GaussianBlur(sp_map,(3,3),0)
        blur_map[self.map==self.colors['black']] = sp_map[self.map==self.colors['black']]
        # add s&p noise to the region surrounding the surrounding the contours
        noised_map = skimage.util.random_noise(blur_map,'s&p',amount=np.random.uniform(0.1+level*0.1,0.2+level*0.1),salt_vs_pepper=np.random.uniform(0.4,0.6))
        # use Gaussian noise and blurring to increase the diversity of noisy pixels' intensities
        noised_map = skimage.util.random_noise(noised_map,var =np.random.uniform(0.01+level*0.02,0.03+level*0.02))
        noised_map = np.round(noised_map*255).astype(np.uint8)
        noised_map[blur_map==self.colors['gray']]=self.colors['gray']
        noised_map[blur_map==self.colors['white']]=self.colors['white']
        noised_map = cv2.GaussianBlur(noised_map,(3,3),0)
        # remain a part of contour as pure black pixels (unoised)
        noised_map[blur_map==self.colors['black']] = self.colors['black']
        # add pepper noise to the unoccupied region
        canvas = np.ones_like(noised_map)*254
        canvas = skimage.util.random_noise(canvas,'pepper',amount=np.random.uniform(0.005+level*0.005,0.01+level*0.005))
        canvas = np.round(canvas*255)
        noised_map[noised_map==self.colors['white']]=canvas[noised_map==self.colors['white']]
        #rotate the map and weaken the added pepper noise
        rotate_kernel = cv2.getRotationMatrix2D(tuple(self.center),rotate_angle,1)
        rotate_map = cv2.warpAffine(noised_map,rotate_kernel,self.img_size,borderValue=self.colors['gray'])
        # add pepper noise one more time
        canvas = np.ones_like(rotate_map)*254
        canvas = skimage.util.random_noise(canvas,'pepper',amount=np.random.uniform(0.005+level*0.005,0.01+level*0.005))
        canvas = np.round(canvas*255)
        rotate_map[rotate_map==self.colors['white']]=canvas[rotate_map==self.colors['white']]
        
        return rotate_map
    
    def spNoise(self,rotate_angle,level):
        # add salt noise to the contours of rooms and furnitures
        sp_map = skimage.util.random_noise(self.map,'salt',amount=np.random.uniform(0.05+level*0.1,0.15+level*0.1))
        sp_map = np.round(sp_map*255).astype(np.uint8)
        sp_map[self.map==self.colors['gray']]=self.colors['gray']
        sp_map[self.map==self.colors['white']]=self.colors['white']
        # find the region surrounding the contours
        blur_map = cv2.GaussianBlur(sp_map,(3,3),0)
        blur_map[self.map==self.colors['black']] = sp_map[self.map==self.colors['black']]
        # add s&p noise to the region surrounding the surrounding the contours
        noised_map = skimage.util.random_noise(blur_map,'s&p',amount=np.random.uniform(0.1+level*0.1,0.2+level*0.1),salt_vs_pepper=np.random.uniform(0.4,0.6))
        # use Gaussian noise and blurring to increase the diversity of noisy pixels' intensities
        noised_map = skimage.util.random_noise(noised_map,var =np.random.uniform(0.01+level*0.02,0.03+level*0.02))
        noised_map = np.round(noised_map*255).astype(np.uint8)
        noised_map[blur_map==self.colors['gray']]=self.colors['gray']
        noised_map[blur_map==self.colors['white']]=self.colors['white']
        noised_map = cv2.GaussianBlur(noised_map,(3,3),0)
        # remain a part of contour as pure black pixels (unoised)
        noised_map[blur_map==self.colors['black']] = self.colors['black']
        #rotate the map
        rotate_kernel = cv2.getRotationMatrix2D(tuple(self.center),rotate_angle,1)
        rotate_map = cv2.warpAffine(noised_map,rotate_kernel,self.img_size,borderValue=self.colors['gray'])
        
        return rotate_map
    def fullNoise(self,rotate_angle,level):
        # blur the map
        blur_map = cv2.GaussianBlur(self.map,(3,3),0)    
        blur_map[self.map==self.colors['black']] = self.map[self.map==self.colors['black']]
        # rotate the map
        rotate_kernel = cv2.getRotationMatrix2D(tuple(self.center),rotate_angle,1)
        rotate_map = cv2.warpAffine(blur_map,rotate_kernel,self.img_size,borderValue=self.colors['gray'])
        # add strong Gaussian noise to the free and occupied regions
        noised_map = skimage.util.random_noise(rotate_map,var =np.random.uniform(0.01+level*0.02,0.03+level*0.02))
        noised_map = np.round(noised_map*255).astype(np.uint8)
        noised_map[rotate_map==self.colors['gray']]=self.colors['gray']
        # add weak Gaussian noise to the unoccupied region
        noised_map = skimage.util.random_noise(noised_map,var =np.random.uniform(0.001+level*0.001,0.002+level*0.001))
        noised_map = np.round(noised_map*255).astype(np.uint8)
        
        return noised_map
        
    def noNoise(self,rotate_angle,level):
        # no noise, not rotate the map
        rotate_kernel = cv2.getRotationMatrix2D(tuple(self.center),rotate_angle,1)
        rotate_map = cv2.warpAffine(self.map,rotate_kernel,self.img_size,borderValue=self.colors['gray'])
        
        return rotate_map
    
    def add_noise(self,noise_type,rotate_angle,level):
        """
        add the noise to the map according to the inputs
        """
        noise_function = getattr(self, noise_type)        
        rotate_map =noise_function(rotate_angle,level)
    
        return rotate_map

if __name__=='__main__': 
    map_num=1
    noise = 'multiNoise'
    level=1    
    map_sim = orig_map.orig_map()
    maps,masks = map_sim.creater(map_num)
    for single_map in maps:
        rotate_angle = np.random.randint(0,360)
        addnoise = addNoise(single_map)
        noise_map = addnoise.add_noise(noise,rotate_angle,level)
        cv2.imshow('map',noise_map)
        cv2.waitKey()
        cv2.destroyAllWindows()