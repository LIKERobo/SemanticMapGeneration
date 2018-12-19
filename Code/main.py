#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main function

Created on Wed Dec 12 17:09:17 2018

@author: chen
"""
import numpy as np
from orig_map import orig_map
from addNoise import addNoise
from dataExtraction import dataExtraction

def main(map_num,noise_types,noise_levels,mode):
    """
    call the classes to generate and label the dataset automatically
    map_num: the number of simulated maps
    noise_types: the types of added noise
    noise_levels: the levels of added noise
    mode: 0-background, 1-doorway
    """
    map_sim = orig_map()
    maps,masks = map_sim.creater(map_num)
    extract = dataExtraction()
    for single_map,single_mask in zip(maps,masks):
        noised_maps = []
        rotate_angle = np.random.randint(0,360)
        addnoise = addNoise(single_map)
        ref_map = addnoise.add_noise('noNoise',rotate_angle,0)
        for noise in noise_types:
            multilevels_maps = []
            for l in noise_levels:    
                noise_map = addnoise.add_noise(noise,rotate_angle,l)
                multilevels_maps.append(noise_map)
            noised_maps.append(multilevels_maps)
        extract.data(noised_maps,ref_map,single_mask,rotate_angle,mode)
        
if __name__=='__main__': 
    noise_types = ['multiNoise','fullNoise']
    noise_levels = [1,4,7]
    main(10,noise_types,noise_levels,1)
    