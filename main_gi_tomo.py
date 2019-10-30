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
    recon_all = get_recon(sino_dict, fignum=40, algorithms = ['gridrec', 'fbp'])

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

# Alloc array
proj = copy.deepcopy(data_sort[peak]).values
sino_alldm = np.zeros([len(theta),  int(len(proj)/len(theta)), len(list_peaks)])
sino_dm =  np.zeros([len(theta),  int(len(proj)/len(theta))])

# Get sino for a domain    
ori_angle = 30*2
flag_normalize = 0
width = 0
for ii, peak in enumerate(list_peaks_sorted[0:2]):
    proj = copy.deepcopy(data_sort[peak]).values #- data_sort['sumBKG0'].values*4
    proj = np.reshape(proj, (len(theta), 1, int(len(proj)/len(theta))) )

    print(proj[bkg_max_idx])
    proj = proj - proj_bkg*proj[bkg_max_idx]
    #proj = proj[:,:,6:]
    proj = np.squeeze(proj)
    if flag_normalize:
        proj = proj - np.min(proj)
        proj = proj / np.max(proj)
    sino_alldm[:,:,ii] = proj[ori_angle,:]
    angle =  ori_angle + peak_angles_orig[ii]
    print('angle = {}'.format(angle))
    #angle = (angle-180) if angle>180 else angle
    sino_dm[int(angle),:] = get_sino_line(proj,  angle, width) #proj[int(angle),:] 
    
# Tomo recon
proj_dm = np.reshape(sino_dm, [sino_dm.shape[0], 1, sino_dm.shape[1]])
algo = 'fbp'
rot_center = tomopy.find_center(proj_dm, theta, init=cen_init, ind=0, tol=0.1)
recon = tomopy.recon(proj_dm, theta, center=rot_center, algorithm=algo)
recon = tomopy.circ_mask(recon, axis=0, ratio=0.95)
    
# Plot sino
plt.figure(55); plt.clf()
plt.subplot(121)
plt.imshow(np.log10(sino_dm[:,:]), cmap='jet', aspect='auto', extent = [axis_x[0], axis_x[-1], theta[-1], theta[0]], vmin=0, vmax=5)
plt.colorbar()

plt.subplot(122)
plt.imshow(recon[0, :,:], cmap='jet') #, vmin=0, vmax=1.5e7)
plt.title(algo)
plt.colorbar()


    
    
    
    
    
    
    