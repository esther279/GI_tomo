#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time, os, sys, re, glob, random, copy
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import signal
from scipy import misc
import tomopy

import analysis.tomo as tomo

## Input
if 1:
    fn = './img/sample1.png';  
    flag_input_png = True
    rot_angles = [0]
else:
    fn = '../results_tomo/domains_recon.npy'; 
    flag_input_png = False
    rot_angles = np.load('../results_tomo/rot_angles.npy')

flag_generate_sino = 1
flag_save_png = 0
out_dir = '../results_tomo/'

## Load pattern
if flag_input_png:
    temp = Image.open(fn).convert('L')
    temp = np.asarray(temp)
    img = temp.copy(); img.setflags(write=1)
    #kernal = np.array([[0.25, 0.5, 0.25],
    #                   [0.5, 1, 0.5],
    #                   [0.25, 0.5, 0.25]]) 
    #kernal = kernal/np.sum(kernal)
    #img = signal.convolve2d(img, kernal, boundary='symm', mode='same')
    
    img = 255-img
    img = np.around(img/np.max(img)*100)
    temp_angle = [43, 63, 87, 100] 
else:
    img = np.load(fn)+56
    temp_angle = np.unique(img[~np.isnan(img)])
    print('domain angles = {}'.format(temp_angle))

            
# Plot
plt.figure(1, figsize=[8,18]); plt.clf()
#plt.subplot(311)
plt.imshow(img, cmap='summer'); plt.colorbar()
plt.title('domain angles = {}'.format(temp_angle))
        

### Generate sino for a peak
if flag_generate_sino:
    t0 = time.time()
    
    th_1 = 0  #***Specify. np.min([np.nanmin(img), 0]) - 1
    th_2 = 180
    th_step = 2
    
    Nproj = len(rot_angles)
    plt.figure(2, figsize=[22,12]); plt.clf()
    
    sino_peak = []
    rot_angles = [0]
    for aa, origin in enumerate(rot_angles): #np.arange(0,360,15):
        thetas_deg = np.arange(th_1+origin, th_2+origin, th_step)
        thetas_rad = thetas_deg/180*np.pi
    
        sino_peak.append([])
        for ii, theta in enumerate(thetas_deg):
            if ii%5==0: print('theta={:.2f}'.format(theta))
            img_domain = img.copy()
            mask = (img_domain==theta-origin)
            img_domain[mask] = 1
            img_domain[~mask] = 0
            img_3d = img_domain.reshape(1, img.shape[0], img.shape[1])
            proj1d = np.squeeze(tomopy.project(img_3d, thetas_rad[ii], center=None, emission=True, pad=True))
            sino_peak[aa].append(proj1d)
        print('time {}'.format(time.time()-t0))            
            
        # Plot
        plt.subplot(1, Nproj, aa+1)
        plt.imshow(np.log10(sino_peak[aa]), aspect='auto')
        #plt.imshow(np.log10(sino_peak[aa]), aspect='auto', extent=[1, len(sino_peak[0]), np.max(thetas_deg), np.min(thetas_deg)]); 
        #plt.colorbar()
        #plt.ylabel('degree')
        plt.title('{}'.format(origin))
        plt.show()
        
        
        # Save to png
        if flag_save_png:
            fn_out = out_dir+'sim_sino'
            fn_out = check_file_exist(fn_out)
            plt.savefig(fn_out, format='png')
        
        
    ## Plot only
    plt.figure(2, figsize=[22,12]); plt.clf()
    for aa, origin in enumerate(rot_angles):
        thetas_deg = np.arange(th_1+origin, th_2+origin, th_step)
        plt.subplot(1, Nproj, aa+1)
        plt.imshow(np.log10(sino_peak[aa]), aspect='auto', extent=[1, len(sino_peak[0]), np.max(thetas_deg), np.min(thetas_deg)]); 
        plt.title('{}'.format(origin))

    tomo.plot_sino(sino_peak[aa], fignum=3, theta=thetas_deg)
        
        