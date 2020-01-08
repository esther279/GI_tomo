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


## Load pattern
fn = './img/exp_sample2.png'
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

ori_angle = 100 # 43, 63, 87, 100
img[img!=ori_angle] = 0


plt.figure(10, figsize=[8,18]); plt.clf()
plt.subplot(311)
plt.imshow(img, cmap='gray')
plt.colorbar()
plt.title('{}, {}'.format(fn, ori_angle))


### Generate sino
img_3d = img.reshape(1,img.shape[0], img.shape[1])
thetas = tomopy.angles(180, 0, 360)
#thetas = tomopy.angles(180, 0+ori_angle, 180+ori_angle)
sino = tomopy.project(img_3d, thetas, center=None, emission=True, pad=True)

plt.subplot(312)
plt.imshow(sino[:,0,:], aspect='auto'); plt.colorbar()


### Tomo recon
rot_center = tomopy.find_center(sino, thetas, init=552, ind=0, tol=0.1)
print(rot_center)
algo = 'fbp' #'gridrec'
recon = tomopy.recon(sino, thetas, center=rot_center, algorithm=algo)
recon = tomopy.circ_mask(recon, axis=0, ratio=0.95)
plt.subplot(313)
plt.imshow(recon[0, :,:], cmap='gray') #, vmin=0, vmax=100)
plt.colorbar()
plt.title(algo)


### Limited angles
peak_angles = ori_angle + np.asarray([0, 20.1, 36.1, 55.6, 90, 180-20.1, 180-36.1, 180-55.6])
peak_angles = [(x-180) if x>180 else x for x in peak_angles]
temp_angles = np.arange(0, 181, 1)
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

plt.figure(11, figsize=[8,18]); plt.clf()
plt.subplot(312)
plt.imshow(sino_la[:,0,:], aspect='auto')# , extent=[0, sino.shape[2], 0, 180]); 
plt.colorbar()

plt.subplot(313)
plt.imshow(recon_la[0, :,:], cmap='gray') #, vmin=0, vmax=100)
plt.colorbar()
plt.title(algo)







