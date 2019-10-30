#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, glob, time, sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import tomopy
from joblib import Parallel, delayed
from scipy import signal
from fun_peaks import *
from fun_tomo_recon import *


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
flag_get_peaks = 0;  flag_LinearSubBKG = 0
flag_load_peaks = 1
flag_tomo = 1

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
        plt.imshow(np.log10(temp2), vmin=0.6, vmax=2.5); plt.colorbar()    
        get_peaks(infiles[0], verbose=2)
        
        fn_out = out_dir+filename+'_peak_roi'
        fn_out = check_file_exist(fn_out)
        plt.savefig(fn_out, format='png')
    
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
        results = parallel( delayed(get_peaks)(infile, verbose=1, flag_LinearSubBKG=flag_LinearSubBKG) for infile in infiles )
    print("\nLoad data and define peak roi: {:.0f} s".format(time.time()-t0))
    
    
    # Pass to pd
    df_peaks = pd.DataFrame()
    for ii, df in enumerate(results):
        df_peaks = df_peaks.append(df, ignore_index=True)
    print(df_peaks)
    print(df_peaks.columns)
    
    # Save 
    fn_out = 'df_peaks_all_subbgk{}'.format(flag_LinearSubBKG)
    fn_out = check_file_exist(fn_out)
    df_peaks.to_csv(fn_out)
 

##########################################
# Sino and recon
########################################## 
if flag_load_peaks:
    df_peaks = pd.read_csv('df_peaks_all_subbg')

## Create sino from pd data
#list_peaks = ['sum12L','sum12Lb']
data_sort, sino_dict = get_sino_from_data(df_peaks, list_peaks=[], flag_rm_expbg=1, flag_thr=0)
print(sino_dict['list_peaks'])
sino_sum = get_sino_sum(sino_dict)

## Plot sino
plot_sino(sino_dict, fignum=30, filename=filename, vlog10=[0, 4])

## Do and plot recon
if flag_tomo:
    recon_all = get_recon(sino_dict, algorithms = ['gridrec', 'fbp'], fignum=40)

    fn_out = out_dir+filename+'peaks_sino_tomo_subbg'+str(flag_LinearSubBKG); 
    fn_out = check_file_exist(fn_out)
    plt.savefig(fn_out, format='png')



##########################################
# Create sino for a domain    
##########################################
#peak_angles_orig =  np.asarray([0, 20.1, 36.1, 55.6, 90, 180-55.6, 180-36.1, 180-20.1, 180])
peak_angles_orig =  np.asarray([0, 20, 30, 51, 90, 51+65, 30+104, 20+140, 180]*2)
list_peaks_sorted = ['sum20L', 'sum21L', 'sum11L', 'sum12L',  'sum02L']
for ii in np.arange(0,4):
    list_peaks_sorted.append(list_peaks_sorted[3-ii])

data_sort_dm, sino_dict_dm = get_sino_from_data(df_peaks, list_peaks=list_peaks_sorted, flag_rm_expbg=1, flag_thr=0)

ori_angle = 16.5 # degree
width = 1 # pixel
sino_allpeaks = sino_dict_dm['sino_allpeaks']
sino_dm = np.zeros([sino_allpeaks.shape[0], sino_allpeaks.shape[1]])
theta = sino_dict_dm['theta']
for ii in np.arange(0, sino_allpeaks.shape[2]):
    sino = sino_allpeaks[:,:,ii]
    peak = list_peaks[ii] if list_peaks!=[] else ''
    
    angle =  ori_angle + peak_angles_orig[ii]
    print('angle = {}'.format(angle))
    angle_idx = get_idx_angle(theta, theta=angle)
    sino_dm[angle_idx,:] = get_proj_from_sino(sino,  angle_idx, width) 

## Plot sino
plot_sino(sino, fignum=50, theta = sino_dict_dm['theta'], axis_x = sino_dict_dm['axis_x'], filename=filename+'.sino_dm', vlog10=[0, 5])
plot_sino(sino_dm, fignum=51, theta = sino_dict_dm['theta'], axis_x = sino_dict_dm['axis_x'], filename=filename+'.sino_dm', vlog10=[0, 5])

# Tomo recon
recon_all = get_recon(sino_dm, theta = sino_dict_dm['theta'], rot_center=27, algorithms = ['gridrec', 'fbp'], fignum=100)



    
    
    
    
    
    
    