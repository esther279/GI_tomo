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
fn = 'sample1.png'
temp = Image.open(fn).convert('L')
temp = np.asarray(temp)
img = temp.copy(); img.setflags(write=1)
#kernal = np.array([[0.25, 0.5, 0.25],
#                   [0.5, 1, 0.5],
#                   [0.25, 0.5, 0.25]]) 
#kernal = kernal/np.sum(kernal)
#img = signal.convolve2d(img, kernal, boundary='symm', mode='same')
img = 255-img
img = img/np.max(img)
img[img<0.99]= 0


plt.figure(10); plt.clf()
plt.subplot(311)
plt.imshow(img, cmap='gray')
plt.colorbar()


### Generate sino
img_3d = img.reshape(1,img.shape[0], img.shape[1])
thetas = tomopy.angles(180, 0, 180)
sino = tomopy.project(img_3d, thetas, center=None, emission=True, pad=True)

plt.subplot(312)
plt.imshow(sino[:,0,:], aspect='auto'); plt.colorbar()


### Tomo recon
rot_center = tomopy.find_center(sino, thetas, init=552, ind=0, tol=0.1)
print(rot_center)
algo = 'gridrec'
recon = tomopy.recon(sino, thetas, center=rot_center, algorithm=algo)
recon = tomopy.circ_mask(recon, axis=0, ratio=0.95)
plt.subplot(313)
plt.imshow(recon[0, :,:], cmap='gray', vmin=0, vmax=1)
plt.colorbar()
plt.title(algo)


### Limited angles
angles = np.arange(0, 180, 1.0)
angles_have = 15.0 + np.round(np.asarray([0, 20.1, 36.1, 55.6, 90, 180-20.1, 180-36.1, 180-55.6]))
angles_na = [int(x) for x in angles if x not in angles_have]
ratio = 1/len(angles_have)*len(angles)

sino_limited = sino.copy()
sino_limited[angles_na] = 0 #temp: assuming angle same as index

recon = tomopy.recon(sino_limited, thetas, center=rot_center, algorithm=algo)
recon = tomopy.circ_mask(recon, axis=0, ratio=0.95)
recon = recon * ratio

plt.figure(11); plt.clf()
plt.subplot(312)
plt.imshow(sino_limited[:,0,:], aspect='auto'); plt.colorbar()
plt.subplot(313)
plt.imshow(recon[0, :,:], cmap='gray', vmin=0, vmax=1)
plt.colorbar()
plt.title(algo)


