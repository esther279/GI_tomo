#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 12:08:47 2019

@author: etsai
"""

import time, os, sys, re, glob, random, copy
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from scipy import signal
from scipy import misc

import tomopy


## Input
if 0:
    fn = './img/exp_sample2.png';  
    flag_input_png = True
    rot_angles = [0]
else:
    fn = '../results_tomo/domains_recon.npy'; 
    flag_input_png = False
    rot_angles = np.load('../results_tomo/rot_angles.npy')
    sino_all_list = np.load('../results_tomo/sino_all_list.npy')

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
    
    
plt.figure(10, figsize=[18,18]); plt.clf()
plt.subplot(331)
plt.imshow(img, cmap='gray')
plt.colorbar()
plt.title('{}, {}'.format(fn, temp_angle))


### Generate sino
img_3d = img.reshape(1,img.shape[0], img.shape[1])
thetas = tomopy.angles(720, 0, 359)
#thetas = tomopy.angles(180, 0+ori_angle, 180+ori_angle)
sino = tomopy.project(img_3d, thetas, center=None, emission=True, pad=True)

plt.subplot(332)
plt.imshow(sino[:,0,:], aspect='auto'); plt.colorbar()

### Tomo recon
rot_center = tomopy.find_center(sino, thetas, init=len(img)/2, ind=0, tol=0.1)
rot_center =  37
#print(rot_center)
algo = 'fbp' #'gridrec'
recon = tomopy.recon(sino, thetas, center=rot_center, algorithm=algo)
recon = tomopy.circ_mask(recon, axis=0, ratio=0.95)
plt.subplot(333)
plt.imshow(recon[0, :,:], cmap='gray') #, vmin=0, vmax=100)
plt.colorbar()
plt.title(algo)


### Limited angles
peak_angles = np.asarray([0, 20.1, 36.1, 55.6, 90, 180-20.1, 180-36.1, 180-55.6])
peak_angles = [(x-180) if x>180 else x for x in peak_angles]
temp_angles = np.arange(0, 180, 1)
ratio = len(peak_angles) / len(temp_angles)
angles_all = np.append(peak_angles, temp_angles)
angles_all.sort()

thetas_la = angles_all / 180 * np.pi
sino_la = tomopy.project(img_3d, thetas_la, center=None, emission=True, pad=True)
for ii, a in enumerate(angles_all):
    if a in temp_angles and a not in peak_angles:
        sino_la[ii,0,:] = sino_la[ii,0,:]*0

recon_la = tomopy.recon(sino_la, thetas_la, center=rot_center, algorithm=algo)
recon_la = tomopy.circ_mask(recon_la, axis=0, ratio=0.95)
recon_la = recon_la / ratio

plt.subplot(335)
plt.imshow(sino_la[:,0,:], aspect='auto')# , extent=[0, sino.shape[2], 0, 180]); 
plt.colorbar()

plt.subplot(336)
plt.imshow(recon_la[0, :,:], cmap='gray') #, vmin=0, vmax=100)
plt.colorbar()
plt.title(algo)


####
sino_exp = sino_all_list[7]
sino_exp = sino_exp.reshape(sino_exp.shape[0], 1, sino_exp.shape[1])
recon_exp = tomopy.recon(sino_exp, thetas, center=30, algorithm=algo)
recon_exp = tomopy.circ_mask(recon_exp, axis=0, ratio=0.95)

plt.subplot(338)
plt.imshow(sino_exp[:,0,:], aspect='auto')
plt.colorbar()

plt.subplot(339)
plt.imshow(recon_exp[0, :,:], cmap='gray')
plt.colorbar()











