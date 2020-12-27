#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, glob, time, sys
import numpy as np
import matplotlib.pyplot as plt
import copy


def plot_quiver(angle):
    angle = angle+90
    N, M = angle.shape
    X, Y = np.meshgrid(np.arange(0, N), np.arange(0, M))
    U = np.zeros([N,M])
    V = np.zeros([N,M])
    for ii in np.arange(0,N):
      for jj in np.arange(0,M):
        U[ii,jj] = np.sin(angle[ii,jj]/180*np.pi)
        V[ii,jj] = np.cos(angle[ii,jj]/180*np.pi)
    
    plt.quiver(X, Y, V, U , units='width', headwidth =0, minshaft=0.05 , minlength=0.5, width=0.005, color='w')


# =============================================================================
# Plot a box
# =============================================================================
def plot_box(center, size, color='r'):
    x_start = center[0] - int(size[0]/2)
    x_end = center[0] + int(size[0]/2)
    y_start = center[1] - int(size[1]/2)
    y_end = center[1] +int(size[1]/2)
    plt.plot([y_start, y_end], [x_start, x_start], color=color)
    plt.plot([y_start, y_end], [x_end, x_end], color=color)
    plt.plot([y_start, y_start], [x_start, x_end], color=color)
    plt.plot([y_end, y_end], [x_start, x_end], color=color)
    
    
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
   
    
    
    
