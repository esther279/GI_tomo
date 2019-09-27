#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, glob, time, sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import tomopy
from joblib import Parallel, delayed
from fun_peaks import *

# /home/etsai/BNL/Research/GIWAXS_tomo_2019C3/RLi/waxs/GI_tomo

########################################## 
# Specify input
##########################################
source_dir = '../raw/'
out_dir = './figs/'
infiles = glob.glob(os.path.join(source_dir, '*tomo_real_*.tiff'))
#for ii in [2,3]: infiles.extend(glob.glob(os.path.join(source_dir, '*tomo_real_*00{}*.tiff'.format(ii))))
filename = infiles[0][infiles[0].find('raw')+4:infiles[0].find('real_')+5]
N_files = len(infiles)
print('N_files = {}'.format(N_files))
# e.g. ../raw/C8BTBT_0.1Cmin_tomo_real_9_x-3.600_th0.090_1.00s_2526493_000656_waxs.tiff'

flag_load_raw_data = 0
flag_get_peaks = 0
flag_load_peaks = 1
flag_sum_peaks = False
flag_tomo = 0

########################################## 
# Load all data and plot sum
##########################################
if flag_load_raw_data:
    t0 = time.time()   
    for ii, infile in enumerate(infiles):
        print("{}/{}, {}".format(ii, N_files, infile))
        temp = Image.open(infile).convert('I')
        data = np.copy(np.asarray(temp))
        if ii==0:
            data_sum = data
        else:
            data_sum = data_sum+data
    data_avg = data_sum/np.size(infiles)    
    print("Data loading: {:.0f} s".format(time.time()-t0))
        
    # Plot
    plt.figure(1); plt.clf()
    plt.imshow(np.log10(data_avg), vmin=0.6, vmax=1)
    plt.colorbar()
    plt.title('Average over {} data \n {}'.format(N_files,infiles[0]))

    # Save as npy
    fn_out = 'data_avg.npy'
    np.save(fn_out, data_avg)
    #### Load and plot to define roi
    if True:
        temp2 = np.load(fn_out)
        plt.figure(100, figsize=[12,12]); plt.clf(); plt.title(fn_out)
        plt.imshow(np.log10(temp2), vmin=0.6, vmax=1.2); plt.colorbar()    
        get_peaks(infiles[0], verbose=2)
        plt.savefig(out_dir+filename+'_peak_roi', format='png')
    
    # Save as tiff
    if False:
        final_img = Image.fromarray((data_avg).astype(np.uint32))
        infile_done = 'CBTBT_0.1Cmin_tomo_real_data_avg.tiff'
        final_img.save(infile_done)     
        
        # Load and plot
        temp = Image.open(infile_done).convert('I') # 'I' : 32-bit integer pixels
        data_avg_infile = np.copy( np.asarray(temp) )
        plt.figure(2); plt.clf()
        plt.imshow(data_avg_infile)
        plt.clim(0, 20)
        plt.colorbar()
        plt.show()
        

##########################################
# Get peaks
##########################################
if flag_get_peaks:
    t0 = time.time()
    with Parallel(n_jobs=4) as parallel:
        results = parallel( delayed(get_peaks)(infile, verbose=1, flag_LinearSubBKG=1) for infile in infiles )
    print("\nLoad data and define peak roi: {:.0f} s".format(time.time()-t0))
    
    
    # Pass to pd
    df_peaks = pd.DataFrame()
    for ii, df in enumerate(results):
        df_peaks = df_peaks.append(df, ignore_index=True)
    print(df_peaks)
    print(df_peaks.columns)
    
    # Save 
    df_peaks.to_csv('df_peaks_all_subbgk_1')
 
##########################################
# Plot Sino
##########################################   
if flag_load_peaks:
    df_peaks = pd.read_csv('df_peaks_all_subbgk_1')
data_sort = df_peaks.sort_values(by=['pos_phi', 'pos_x'])
#data_sort_drop = data_002_sort[data_sort.pos_phi >=0]

theta = data_sort['pos_phi']
theta = theta.drop_duplicates()
theta = np.asarray(theta)

axis_x = data_sort['pos_x']
axis_x = axis_x.drop_duplicates()
axis_x = np.asarray(axis_x)
   

##########################################
# Create projection from pd data
##########################################   
# Sum over peak sinos
if flag_sum_peaks:
    for ii, peak in enumerate(list(df_peaks.columns[6:-2])):
        print(peak)
        proj_orig = data_sort[peak] # + data_sort['sum11L'] 
        proj = proj_orig.values 
        proj = np.reshape(proj, (len(theta), 1, int(len(proj)/len(theta))) )      
        
        proj = proj[:,:,6:]
        thr = np.mean(proj)*0.5
        print('thr = {}'.format(thr))
        proj[proj<thr] = 0
        proj[150:420,:,:] = 0
        if ii==0:
            proj_sum = proj
        else:
            proj_sum = proj_sum + proj

    proj = proj_sum
    

# Create projection from pd data and make tomo 
plt.figure(30, figsize=[20,15]); plt.clf()
for ii, peak in enumerate(list(df_peaks.columns[5:])):
#if 1:
#    peak = 'sum20L'
    proj_orig = data_sort[peak] # + data_sort['sum11L'] 
    proj = proj_orig.values
    proj = np.reshape(proj, (len(theta), 1, int(len(proj)/len(theta))) )
    proj = proj[:,:,6:]
    #proj = pow(proj,1.2)
    
    #thr = np.mean(proj)*1.5
    #print('thr = {}'.format(thr))
    #proj[proj<thr] = 0
    #proj[150:550,:,:] = 0
    
    # Plot
    plt.figure(30)
    plt.subplot(3,5,ii+1)
    plt.imshow(np.log10(proj[:,0,:]), cmap='jet', aspect='auto', extent = [axis_x[0], axis_x[-1], theta[-1], theta[0]])
    plt.axis('off')
    if ii==0: 
        plt.title('{}\n{}'.format(filename, peak), fontweight='bold')
        plt.xlabel('pos_x (mm)')
        plt.ylabel('pos_phi (deg)')    
        plt.axis('on')
    elif ii%2: plt.title('{}'.format(peak), fontweight='bold')
    else: plt.title('{}'.format(peak))

    #plt.subplot(5,13,13+ii+1)
    #plt.imshow(np.log10(proj[:,0,:]), cmap='jet', aspect='auto', extent = [axis_x[0], axis_x[-1], theta[-1], theta[0]])
    #plt.colorbar(); plt.axis('off')
 
    
    ##########################################
    # Find center
    ##########################################   
    if flag_tomo:
        #proj = tomopy.minus_log(proj)
        flat = np.ones((1,1,proj.shape[2]))
        dark = np.zeros((1,1,proj.shape[2]))
        proj = tomopy.normalize(proj, flat, dark) # (proj-dark)/(flat-dark)
        
        #rot_center = tomopy.find_center(proj, theta, init=290, ind=0, tol=0.5)
        cen_init = proj.shape[2]/2
        rot_center = tomopy.find_center(proj, theta, init=cen_init, ind=0, tol=0.1)
        print('Rotational center: {}'.format(rot_center))
        if (rot_center>cen_init+5) or (rot_center<cen_init-5):
            rot_center = proj.shape[2]/2
            
        
        ##########################################
        # Tomo reconstruction
        ##########################################      
        # Keyword "algorithm" must be one of ['art', 'bart', 'fbp', 'gridrec', 'mlem', 'osem', 'ospml_hybrid', 
        # 'ospml_quad', 'pml_hybrid', 'pml_quad', 'sirt', 'tv', 'grad'], or a Python method.
        
        #recon = tomopy.recon(proj, theta, center=rot_center, algorithm=tomopy.lprec, lpmethod='tv', ncore=1, num_iter=512, reg_par=5e-4)
        
        rot_center = 23#cen_init
        algorithms = ['fbp', 'mlem', 'tv']
        #plt.figure(50)
        for jj, algo in enumerate(algorithms):
            recon = tomopy.recon(proj, theta, center=rot_center, algorithm=algo)
            recon = tomopy.circ_mask(recon, axis=0, ratio=0.95)
            plt.subplot(5, 13, (jj+2)*13+ii+1)
            plt.imshow(recon[0, :,:], cmap='jet') #, vmin=0, vmax=1.5e7)
            if ii%2: plt.title('{}\n{}, cen{:.1f}'.format(peak, algo, float(rot_center)), fontweight='bold')
            else: plt.title('{}\n{}, cen{:.1f}'.format(peak, algo, float(rot_center)))
            #plt.colorbar(); plt.show();
        
plt.savefig(out_dir+filename+'peaks_sino', format='png')

    
        
        
        
    
    
    
    
    
    
    
    
    