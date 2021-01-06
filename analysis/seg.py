#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, glob, time, sys
import numpy as np
import matplotlib.pyplot as plt
import copy
import tomopy
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


# =============================================================================
# 
# =============================================================================
def do_seg_sino(sino, thetas, thr_range = [0.5, 1.0, 0.05], rot_center=25, algo='fbp'):

    thetas = thetas/180*np.pi
    thr_array = np.arange(thr_range[0], thr_range[1]+thr_range[2], thr_range[2])
    corr_array = np.zeros(len(thr_array))    

    for ii, thr in enumerate(thr_array):
        
        #----- Get recon from original sino -----
        temp_recon = tomopy.recon(sino.reshape(sino.shape[0],1,sino.shape[1]), thetas, center=rot_center, algorithm=algo)
        temp_recon = tomopy.circ_mask(temp_recon, axis=0, ratio=0.95)
        temp_recon = temp_recon/np.max(temp_recon)   
        temp_recon[temp_recon<thr] = 0
    
        #----- Get updated sino -----
        temp_sino = tomopy.project(temp_recon.reshape(1,sino.shape[1],sino.shape[1]), thetas, pad=False)   

        #----- Calc err -----
        err, corr = compare_img(temp_sino, sino)
        corr_array[ii] = corr
        if ii>0:
            print("[{}] thr={:.2f}, err = {:.3f}, corr = {:.3f}".format(ii, thr, err, corr))

        #----- Adjust sino?
        #temp_sino = temp_sino/np.max(temp_sino)
        if 0: 
            temp_sino[temp_sino<=0.5] = 0
            #temp_sino[temp_sino>0.7] = 1
        #temp_sino = temp_sino*sino.reshape(721,1,50)          

        
    idx = np.argmax(corr_array)
    print("thr_array[idx] = {:.3f}".format(thr_array[idx]))
    
    temp_recon = tomopy.recon(sino.reshape(sino.shape[0],1,sino.shape[1]), thetas, center=rot_center, algorithm=algo)
    temp_recon = tomopy.circ_mask(temp_recon, axis=0, ratio=0.95)
    #temp_recon = temp_recon/np.max(temp_recon)
    if 1: #nn<Ns:
        temp_recon[temp_recon<thr_array[idx]*np.max(temp_recon)] = 0
    
    return np.squeeze(temp_recon)
    
    
def compare_img(img1, img2):
    img1 = img1.reshape(-1)
    img2 = img2.reshape(-1)
    err = img1/np.max(img1) - img2/np.max(img2)
    err = np.mean(err**2)
    corr = np.corrcoef(img1, img2)
    corr = corr[0][1]
    
    return err, corr



