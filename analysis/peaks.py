#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, glob, time, sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import tomopy
import copy

import analysis.util as util

# =============================================================================
# Crop array
# =============================================================================
def ArrayCrop(data=None, center=None, size=None):
    #data = img
    if np.size(size) == 1:
        size = [size, size]
    x_start = center[0] - int(size[0]/2)
    x_end = center[0] + int(size[0]/2)
    y_start = center[1] - int(size[1]/2)
    y_end = center[1] +int(size[1]/2)
    return data[x_start:x_end, y_start:y_end]

# =============================================================================
# Substract background
# =============================================================================
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

# =============================================================================
# Substract background
# =============================================================================
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
   
# =============================================================================
# Get peak intensity (based on specified ROI) from raw data (tiff files)    
# =============================================================================
def get_peaks(infile, peak_list, phi_max=360, a=1, verbose = 0, flag_LinearSubBKG = 0, FS=12):
    if verbose>0: print(infile)        
    if verbose>1: print('Parse param manually for now..\n')
    
    temp = infile.split('_')
    a = 0
    zigzag_n = int(temp[3+a])
    pos_x = float(temp[4+a][1:])
    scan_n = int(temp[7+a])
    if zigzag_n%2==0:
        pos_phi = phi_max-float(temp[8+a])/2.0
    else:
        pos_phi = float(temp[8+a])/2.0

    df = pd.DataFrame({'pos_phi':pos_phi,
                   'pos_x':pos_x,
                    'zigzag_n':zigzag_n,
                    'scan_n': scan_n}, index = [pos_phi])
        
    ### Read data
    temp1 = Image.open(infile).convert('I')
    data_infile = np.copy(np.asarray(temp1))
    if verbose>2:
        plt.figure(99); plt.clf()
        #plt.pcolormesh(np.log10(data_infile)) #,vmin=0.1,vmax=2.2); 
        plt.imshow(np.log10(data_infile))      
     
    for p in peak_list:
        N_regions = (len(p)-1)/2 # number of regions for this peak
        peak = p[-1]
        peakarea_sum = 0
        for ii in np.arange(0, N_regions):
            center = p[int(ii*2)]
            size = p[int(ii*2+1)]
            if verbose>1: 
                util.plot_box(center, size, color=[0.9, 0.9, 0.9]) 
                if str(peak[3:])[0]=='B':
                    tt = -60
                else: tt=0
                plt.text(center[1]-5+tt, center[0]+10*np.random.rand(), str(peak[3:]), color='r', fontsize=FS, fontweight='bold')
            
            peakarea = ArrayCrop(data=data_infile, center=center, size=size) 
            if flag_LinearSubBKG:
                [peakarea, BKG] = LinearSubBKG(peakarea)
            peakarea_sum = peakarea_sum + np.sum(peakarea)          
        
        df[str(peak)] = peakarea_sum    

    return df


# =============================================================================
# Calculate ROI area 
# =============================================================================
def calc_area_peakROI(peak_list):
    areas = np.zeros(len(peak_list))
    for ii, temp in enumerate(peak_list):
        roi = temp[1]
        areas[ii] = roi[0]*roi[1]
    return areas
        
        
        
        
        
    