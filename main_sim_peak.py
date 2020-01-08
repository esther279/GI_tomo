#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time, os, sys, re, glob, random, copy
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from scipy import signal
from scipy import misc

import tomopy


## Input
if 1:
    fn = './img/exp_sample2.png';  
    flag_input_png = True
else:
    fn = '../results_tomo/domains_recon.npy'; 
    flag_input_png = False

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
    img = np.load(fn)
    temp_angle = np.unique(img[~np.isnan(img)])
    print('domain angles = {}'.format(temp_angle))

            
# Plot
plt.figure(1, figsize=[8,18]); plt.clf()
plt.subplot(311)
plt.imshow(img, cmap='summer'); plt.colorbar()
plt.title('domain angles = {}'.format(temp_angle))
        

### Generate sino for a peak
if flag_generate_sino:
    t0 = time.time()
    
    th_1 = 50 # np.min([np.nanmin(img), 0]) - 1
    th_2 = 180
    th_step = 5
    
    for origin in [0]: #np.arange(0,360,15):
        thetas_deg = np.arange(th_1+origin, th_2+origin, th_step)
        thetas_rad = thetas_deg/180*pi
    
        sino_peak = []
        for ii, theta in enumerate(thetas_deg):
            if ii%5==0: print('theta={:.2f}'.format(theta))
            img_domain = img.copy()
            mask = (img_domain==theta-origin)
            img_domain[mask] = 1
            img_domain[~mask] = 0
            img_3d = img_domain.reshape(1, img.shape[0], img.shape[1])
            proj1d = np.squeeze(tomopy.project(img_3d, thetas_rad[ii], center=None, emission=True, pad=True))
            sino_peak.append(proj1d)
        print('time {}'.format(time.time()-t0))
            
            
        # Plot
        if flag_generate_sino:
            plt.subplot(312)
            plt.imshow(sino_peak, aspect='auto', extent=[1, len(sino_peak[0]), np.max(thetas_deg), np.min(thetas_deg)]); 
            #plt.colorbar()
            plt.ylabel('degree')
            plt.title('origin (deg) = {}'.format(origin))
            plt.show()
        
        
        # Save to png
        if flag_save_png:
            fn_out = out_dir+'sim_sino'
            fn_out = check_file_exist(fn_out)
            plt.savefig(fn_out, format='png')
        
        
        

