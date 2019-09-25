#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, glob, time, sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import tomopy


def ArrayCrop(data=None, center=None, size=None):
    #data = img
    if np.size(size) == 1:
        size = [size, size]
    x_start = center[0] - int(size[0]/2)
    x_end = center[0] + int(size[0]/2)
    y_start = center[1] - int(size[1]/2)
    y_end = center[1] +int(size[1]/2)
    return data[x_start:x_end, y_start:y_end]

def plot_box(center, size, color='r'):
    x_start = center[0] - int(size[0]/2)
    x_end = center[0] + int(size[0]/2)
    y_start = center[1] - int(size[1]/2)
    y_end = center[1] +int(size[1]/2)
    plt.plot([y_start, y_end], [x_start, x_start], color=color)
    plt.plot([y_start, y_end], [x_end, x_end], color=color)
    plt.plot([y_start, y_start], [x_start, x_end], color=color)
    plt.plot([y_end, y_end], [x_start, x_end], color=color)

def LinearSubBKG(data):

    height, width = np.shape(data)
    #print(height, width)
    A, B = np.meshgrid(height, width)
    
    #create mask for edge
    mask = np.zeros((height, width)) +1
    mask[1:-1, 1:-1] = 0
    
    #calculation
    C = data*mask
    D = A*mask
    E = B*mask
    BKG = np.zeros((height, width)) 
    
    #
    E1 = 2*(width+height)
    Ex = np.sum(np.sum(D))
    Ey = np.sum(np.sum(E))
    Ez = np.sum(np.sum(C))
    Exx = np.sum(np.sum(D*D))
    Exy = np.sum(np.sum(D*E))
    Eyy = np.sum(np.sum(E*E))
    Exz = np.sum(np.sum(D*data))
    Eyz = np.sum(np.sum(E*data))
    
    M = np.array([E1, Ex, Ey, Ex, Exx, Exy, Ey, Exy, Eyy])
    M = M.reshape(3, 3)
    N = np.array([Ez, Exz, Eyz])
    N = N.reshape(3, 1)
    
    i = np.eye(3, 3)
    R = np.matmul(np.linalg.lstsq(M,i)[0],N)
    #R = np.linalg.solve(M, N)
    #print(R)
    for x in range(height):
        for y in range(width):
            BKG[x,y] = R[0] + R[1]*x + R[2]*y
    data_sub = data - BKG

    return data_sub, BKG

def LinearSubBKG_temp(data):

    height, width = np.shape(data)
    print(height, width)
    A, B = np.meshgrid(height, width)
    
    #create mask for edge
    mask = np.zeros((height, width)) +1
    mask[1:-1, 1:-1] = 0
    
    #calculation
    C = data*mask
    
    BKG = np.sum(np.sum(C))/(2*(height+width))
    data_sub = data - BKG
    return data_sub, BKG
   
    
def get_peaks(infile, verbose=0):
    #df_peaks = pd.DataFrame()
    #print(infile)
    
    temp = infile.split('_')
    pos_x = float(temp[5][1:])
    zigzag_n = int(temp[4])
    scan_n = int(temp[8])
    if zigzag_n%2==0:
        pos_phi = 360-float(temp[9])/2.0
    else:
        pos_phi = float(temp[9])/2.0
        
    # Read data
    temp1 = Image.open(infile).convert('I')
    data_infile = np.copy(np.asarray(temp1))
    
    # Define peak roi
    #ss_002 = ArrayCrop(data=data_infile, center=[575, 479], size=[50, 10])
    #[data, BKG] = LinearSubBKG(ss_002)
    #sum002 = np.sum(data)  
    sum002 = 0


    if verbose>1:
        plt.figure(99); plt.clf()
        #plt.pcolormesh(np.log10(data_infile)) #,vmin=0.1,vmax=2.2); 
        plt.imshow(np.log10(data_infile))
    
    center = [525, 735]; size = [150, 10]
    peakarea = ArrayCrop(data=data_infile, center=center, size=size) 
    [data, BKG] = LinearSubBKG(peakarea)
    if verbose>0: plot_box(center, size)
    sum11L = np.sum(data)      
    
    center=[525, 223]; size=[150, 10]
    peakarea = ArrayCrop(data=data_infile, center=center, size=size) 
    [data, BKG] = LinearSubBKG(peakarea)
    if verbose>0: plot_box(center, size)
    sum1_1L = np.sum(data)  
    
    center=[525, 223]; size=[150, 10]
    peakarea = ArrayCrop(data=data_infile, center=center, size=size) 
    [data, BKG] = LinearSubBKG(peakarea)
    if verbose>0: plot_box(center, size)
    sum02L = np.sum(data)     
            
    
    if 1:
        df = pd.DataFrame({'pos_phi':pos_phi,
                       'pos_x':pos_x,
                        'zigzag_n':zigzag_n,
                        'scan_n': scan_n,
                        'sum002':sum002,
                        'sum11L':sum11L,
                        'sum1_1L':sum1_1L,
                        }, index = [pos_phi])
    
        #df_peaks = df_peaks.append(df, ignore_index=True)
    
    return df




