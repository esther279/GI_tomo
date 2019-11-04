#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, glob, time, sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import tomopy
import copy


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
   
    
def get_peaks(infile, verbose = 0, flag_LinearSubBKG = 0):
    if verbose>0: print(infile)        
    if verbose>1: print('Parse param manually for now..\n')
    
    temp = infile.split('_')
    pos_x = float(temp[4][1:])
    zigzag_n = int(temp[3])
    scan_n = int(temp[7])
    if zigzag_n%2==0:
        pos_phi = 360-float(temp[8])/2.0
    else:
        pos_phi = float(temp[8])/2.0

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

      
    ### Define peak roi  
    peak_list = [
            # center, size, peak
            [[575, 479], [60, 10], 'sum002'],
            # 11L
            [[525, 735], [180, 10], 'sum11L'],
            [[525, 223], [180, 10], 'sum11Lb'],
            # 02L
            [[603, 787-3], [30, 10], 'sum02L'],
            [[603, 172], [30, 10], 'sum02Lb'],
            # 12L
            [[589, 848], [58, 6], 'sum12L'], 
            [[589, 110], [58, 6], 'sum12Lb'],
            # 20L
            [[323-6, 903+2], [60, 15], 'sum20L'],
            [[323, 56], [30, 15], 'sum20Lb'],
            # 21L
            [[280, 936+2], [40, 15], 'sum21L'],
            [[280, 26], [40, 15], 'sum21Lb'],
            # Si
            [[400, 809], [12, 12], 'sumSi'],
            [[400, 151], [12, 12], 'sumSib'],
            # background
            [[560, 440], [30,30], 'sumBKG0'],
            ]
     
    for p in peak_list:
        center = p[0]
        center[1] = center[1]+5 if center[1] <470 else center[1]
        size = p[1]
        peak = p[2]
        if verbose>1: 
            plot_box(center, size) 
            plt.text(center[1], center[0], str(peak), color='r')
        
        peakarea = ArrayCrop(data=data_infile, center=center, size=size) 
        if flag_LinearSubBKG:
            [peakarea, BKG] = LinearSubBKG(peakarea)
        peakarea_sum = np.sum(peakarea)          
        df[str(peak)] = peakarea_sum

    return df


def check_file_exist(fn):
    ii=0  
    fn_out = copy.deepcopy(fn)
    while os.path.exists(fn_out):
        ii = ii+1
        fn_out = fn +'_{:d}'.format(ii)
    print('Saving to {}'.format(fn_out))
    
    return fn_out

