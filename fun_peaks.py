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
        
    ### Read data
    temp1 = Image.open(infile).convert('I')
    data_infile = np.copy(np.asarray(temp1))
    if verbose>1:
        plt.figure(99); plt.clf()
        #plt.pcolormesh(np.log10(data_infile)) #,vmin=0.1,vmax=2.2); 
        plt.imshow(np.log10(data_infile))
        
        
    ### Define peak roi   
    # 00L
    center=[575, 479]; size=[60, 10]
    peakarea = ArrayCrop(data=data_infile, center=center, size=size) 
    [data, BKG] = LinearSubBKG(peakarea)
    if verbose>0: plot_box(center, size)
    sum002 = np.sum(data) 

    # 11L
    center = [525, 735]; size = [180, 10]
    peakarea = ArrayCrop(data=data_infile, center=center, size=size) 
    [data, BKG] = LinearSubBKG(peakarea)
    if verbose>0: plot_box(center, size)
    sum11L = np.sum(data)      
    
    center=[525, 223]; size=[180, 10]
    peakarea = ArrayCrop(data=data_infile, center=center, size=size) 
    [data, BKG] = LinearSubBKG(peakarea)
    if verbose>0: plot_box(center, size)
    sum11Lb = np.sum(data)  
    
    # 02L
    center=[603, 787]; size=[30, 10]
    peakarea = ArrayCrop(data=data_infile, center=center, size=size) 
    [data, BKG] = LinearSubBKG(peakarea)
    if verbose>0: plot_box(center, size)
    sum02L = np.sum(data)    
    
    center=[603, 172]; size=[30, 10]
    peakarea = ArrayCrop(data=data_infile, center=center, size=size) 
    [data, BKG] = LinearSubBKG(peakarea)
    if verbose>0: plot_box(center, size)
    sum02Lb = np.sum(data)    
 
    # 12L
    center=[589, 848]; size=[58, 6]
    peakarea = ArrayCrop(data=data_infile, center=center, size=size) 
    [data, BKG] = LinearSubBKG(peakarea)
    if verbose>0: plot_box(center, size)
    sum12L = np.sum(data)       

    center=[589, 110]; size=[58, 6]
    peakarea = ArrayCrop(data=data_infile, center=center, size=size) 
    [data, BKG] = LinearSubBKG(peakarea)
    if verbose>0: plot_box(center, size)
    sum12Lb = np.sum(data)          

    # 20L
    center=[323, 903]; size=[30, 15]
    peakarea = ArrayCrop(data=data_infile, center=center, size=size) 
    [data, BKG] = LinearSubBKG(peakarea)
    if verbose>0: plot_box(center, size)
    sum20L = np.sum(data)       

    center=[323, 56]; size=[30, 15]
    peakarea = ArrayCrop(data=data_infile, center=center, size=size) 
    [data, BKG] = LinearSubBKG(peakarea)
    if verbose>0: plot_box(center, size)
    sum20Lb = np.sum(data)    

    # 21L
    center=[280, 936]; size=[40, 15]
    peakarea = ArrayCrop(data=data_infile, center=center, size=size) 
    [data, BKG] = LinearSubBKG(peakarea)
    if verbose>0: plot_box(center, size)
    sum21L = np.sum(data)       

    center=[280, 26]; size=[40, 15]
    peakarea = ArrayCrop(data=data_infile, center=center, size=size) 
    [data, BKG] = LinearSubBKG(peakarea)
    if verbose>0: plot_box(center, size)
    sum21Lb = np.sum(data)   

    # Si
    center=[400, 809]; size=[12, 12]
    peakarea = ArrayCrop(data=data_infile, center=center, size=size) 
    [data, BKG] = LinearSubBKG(peakarea)
    if verbose>0: plot_box(center, size)
    sumSi = np.sum(data)       

    center=[400, 151]; size=[12, 12]
    peakarea = ArrayCrop(data=data_infile, center=center, size=size) 
    [data, BKG] = LinearSubBKG(peakarea)
    if verbose>0: plot_box(center, size)
    sumSib = np.sum(data)   

    df = pd.DataFrame({'pos_phi':pos_phi,
                   'pos_x':pos_x,
                    'zigzag_n':zigzag_n,
                    'scan_n': scan_n,
                    'sum002':sum002,
                    'sum11L':sum11L,
                    'sum11Lb':sum11Lb,
                    'sum02L':sum02L,
                    'sum02Lb':sum02Lb,
                    'sum12L':sum12L,
                    'sum12Lb':sum12Lb,
                    'sum20L':sum20L,
                    'sum20Lb':sum20Lb,
                    'sum21L':sum21L,
                    'sum21Lb':sum21Lb,
                    'sumSi':sumSi,
                    'sumSib':sumSib,
                    }, index = [pos_phi])
    
        #df_peaks = df_peaks.append(df, ignore_index=True)
    
    return df




