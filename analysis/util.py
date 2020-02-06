#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, glob, time, sys
import numpy as np
import matplotlib.pyplot as plt
import copy


# =============================================================================
# Check if file exists, append file name with number
# =============================================================================
def check_file_exist(fn):
    ii=0  
    fn_out = copy.deepcopy(fn)
    while os.path.exists(fn_out):
        ii = ii+1
        fn_out = fn +'_{:d}'.format(ii)
    print('Saving to {}'.format(fn_out))
    
    return fn_out


# =============================================================================
# Generate a random color
# =============================================================================
def rand_color(a, b):
    r = b-a
    color = (np.random.random()*r+a, np.random.random()*r+a, np.random.random()*r+a)
    return color
    

# =============================================================================
# Return image stack with RGB channels
# =============================================================================
def image_RGB(image, rgb):
    dim = image.shape
    image_stack = np.zeros([dim[0], dim[1], 3])    
    if 'R' in rgb:
        image_stack[:,:,0] = image
    if 'G' in rgb:
        image_stack[:,:,1] = image     
    if 'B' in rgb:
        image_stack[:,:,2] = image
    if 'W' in rgb:
        image_stack[:,:,0] = image
        image_stack[:,:,1] = image
        image_stack[:,:,2] = image
    if 'C' in rgb:
        image_stack[:,:,1] = image
        image_stack[:,:,2] = image
    if 'M' in rgb:
        image_stack[:,:,0] = image
        image_stack[:,:,2] = image           
    if 'Y' in rgb:
        image_stack[:,:,0] = image
        image_stack[:,:,1] = image          
        
    return image_stack
   
    
    
    
