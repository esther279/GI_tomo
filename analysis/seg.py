#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, glob, time, sys
import numpy as np
import matplotlib.pyplot as plt
import copy
#import tomopy
#from scipy.signal import find_peaks

import skimage.draw as draw
import skimage.segmentation as seg

# =============================================================================
# Apply segmentaion (random_walker) and nomralization
# =============================================================================
def do_segmentation(image, centers, width=1, fignum=10):  

    image_labels = np.zeros(image.shape, dtype=np.uint8)
    dim = image.shape[0]
    ra = int(np.floor(dim/2))
    
    indices = draw.circle_perimeter(ra, ra, ra-1)
    image_labels[indices] = 1
    
    s = np.linspace(0, 2*np.pi, dim)
    for center in centers:
        r = center[0] + width*np.sin(s)
        c = center[1] + width*np.cos(s)
        init = np.array([r, c]).T
        image_labels[init[:, 1].astype(np.int), init[:, 0].astype(np.int)] = 2

    image_segmented = seg.random_walker(image, image_labels) - 1
    image_out = image_segmented*image
    image_out = image_out/np.max(image_out)
    
        
    if fignum>0:
        plt.figure(fignum, figsize=[20,10]); plt.clf()
        plt.subplot(211)
        plt.imshow(image_labels);
        plt.imshow(img, alpha=0.3);
        plt.axis([0, image.shape[1], image.shape[0], 0])
        
        plt.subplot(212)
        plt.imshow(image_segmented);
        plt.axis([0, image.shape[1], image.shape[0], 0])
        
    return image_out

   
# =============================================================================
# Apply threshold and normalization  
# thr: [0, 1]
# =============================================================================
def do_thr(image, thr):  

    image_out = image.copy()
    image_out[image < thr*np.max(image)] = 0
    image_out = image_out/np.max(image_out)
        
    return image_out





