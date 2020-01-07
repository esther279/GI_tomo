#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time, os, sys, re, glob, random, copy
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from scipy import signal
from scipy import misc

import tomopy


## Load pattern
fn = 'sample1b.png'
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

#ori_angle = 63 # 43, 63, 87, 100


### Generate sino for a peak
thetas_deg = np.arange(0, 180, 1)
thetas_rad = thetas_deg/180*pi
sino_peak = []
for ii, theta in enumerate(thetas_deg):
    print('theta={:.2f}'.format(theta))
    img_domain = img.copy()
    img_domain[img_domain!=theta] = 0
    img_3d = img_domain.reshape(1, img.shape[0], img.shape[1])
    proj1d = np.squeeze(tomopy.project(img_3d, thetas_rad[ii], center=None, emission=True, pad=True))
    sino_peak.append(proj1d)
    
    
# Plot
plt.figure(1, figsize=[8,18]); plt.clf()
plt.subplot(311)
plt.imshow(img); plt.colorbar()
plt.subplot(312)
plt.imshow(sino_peak, aspect='auto'); plt.colorbar()







