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

# =============================================================================
# Specify input
# =============================================================================
source_dir = '../../raw/'
out_dir = '../results/'
infiles = glob.glob(os.path.join(source_dir, '*C8BTBT_0.1Cmin_tomo_*.tiff'))
flag_load_raw_data = 0
flag_get_peaks = 0;  flag_LinearSubBKG = 1
flag_load_peaks = 1
flag_tomo = 1
# TOMO_sample3_T1_33_x1.200_th0.120_1.00s_2588547_000692_waxs.tiff 

#for ii in [2,3]: infiles.extend(glob.glob(os.path.join(source_dir, '*tomo_real_*00{}*.tiff'.format(ii))))
# e.g. ../raw/C8BTBT_0.1Cmin_tomo_real_9_x-3.600_th0.090_1.00s_2526493_000656_waxs.tiff'
filename = infiles[0][infiles[0].find('C8BTBT'):infiles[0].find('tomo_')+5]
N_files = len(infiles); print('N_files = {}'.format(N_files))
if os.path.exists(out_dir) is False: os.mkdir(out_dir)

# =============================================================================
# Load all/some data and plot sum
# =============================================================================
if flag_load_raw_data:
    t0 = time.time()   
    for ii, infile in enumerate(infiles):
        if ii%10==0: # Quick checck peak positions
            print("{}/{}, {}".format(ii, N_files, infile))
            temp = Image.open(infile).convert('I')
            data = np.copy(np.asarray(temp))
            if ii==0:
                data_sum = data
            else:
                data_sum = data_sum+data
    data_avg = data_sum/np.size(infiles)*10    
    print("Data loading: {:.0f} s".format(time.time()-t0))
        
    # Plot
    plt.figure(1); plt.clf()
    plt.imshow(np.log10(data_avg), vmin=0.6, vmax=1.5)
    plt.colorbar()
    plt.title('Average over {} data \n {}'.format(N_files,infiles[0]))
    fn_out = out_dir+filename+'_avg'
    fn_out = check_file_exist(fn_out)
    plt.savefig(fn_out, format='png')

    # Save as npy
    fn_out = out_dir+'data_avg'
    fn_out = check_file_exist(fn_out)
    np.save(fn_out, data_avg)
    #### Load and plot to define roi
    if True:
        temp2 = np.load(fn_out+'.npy')
        plt.figure(100, figsize=[12,12]); plt.clf(); plt.title(fn_out)
        plt.imshow(np.log10(temp2), vmin=0.3, vmax=1.5); plt.colorbar()    
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
        
# =============================================================================
# Get peaks from raw tiff files
# =============================================================================
if flag_get_peaks:
    t0 = time.time()
    flag_load_parellel = 0  # Sometimes parallel doesn't work..
    if flag_load_parellel:
        with Parallel(n_jobs=3) as parallel:
            results = parallel( delayed(get_peaks)(infile, verbose=1, flag_LinearSubBKG=flag_LinearSubBKG) for infile in infiles )
    else:
        results = []
        for ii, infile in enumerate(infiles):
            #if ii%10==0:
            temp = get_peaks(infile, verbose=1, flag_LinearSubBKG=flag_LinearSubBKG)
            results.append(temp)
    print("\nLoad data and define peak roi: {:.0f} s".format(time.time()-t0))
    
    
    # Pass to pd
    df_peaks = pd.DataFrame()
    for ii, df in enumerate(results):
        df_peaks = df_peaks.append(df, ignore_index=True)
    print(df_peaks)
    print(df_peaks.columns)
    
    # Save 
    fn_out = out_dir+'df_peaks_all_subbgk{}'.format(flag_LinearSubBKG)
    fn_out = check_file_exist(fn_out)
    df_peaks.to_csv(fn_out)
 
# =============================================================================
# Sino and recon
# =============================================================================
if flag_load_peaks:
    df_peaks = pd.read_csv(out_dir+'df_peaks_all_subbg{}'.format(flag_LinearSubBKG))

## Create sino from pd data
list_peaks = ['sum002',
 'sum11L',
 'sum11Lb',
 'sum02L',
 'sum02Lb',
 'sum12L',
 'sum12Lb',
 'sum20L',
 'sum20Lb',
 'sum21L',
 'sum21Lb',
 'sumBKG0']
list_peaks = []
data_sort, sino_dict = get_sino_from_data(df_peaks, list_peaks=list_peaks, flag_rm_expbg=1, flag_thr=1)
print(sino_dict['list_peaks'])
sino_sum = get_sino_sum(sino_dict)

## Plot sino
plot_sino(sino_dict, fignum=30, title_st=filename, vlog10=[0, 5.5])

fn_out = out_dir+filename+'peaks_sino' 
fn_out = check_file_exist(fn_out)
plt.savefig(fn_out, format='png')
    
## Do and plot recon
if flag_tomo:
    recon_all = get_recon(sino_dict, rot_center=32, algorithms = ['gridrec', 'fbp', 'tv'], fignum=40)

    fn_out = out_dir+filename+'peaks_sino_tomo_subbg'+str(flag_LinearSubBKG); 
    fn_out = check_file_exist(fn_out)
    plt.savefig(fn_out, format='png')

   
# =============================================================================
# Label peak positons (deg) a sino
# =============================================================================
peak =  'sum11L'
sino, sum_sino, theta = get_sino_from_a_peak(sino_dict, peak) # which peak roi
plt.figure(10, figsize=[15, 8]); plt.clf()
plt.plot(theta, sum_sino); #plt.ylim(0, 2e7)
label_peaks(line_x, line_y)
plt.title(peak)


# =============================================================================
# Create sino for a domain        
# =============================================================================
## Specify the angles to include for a certain domain by looking at the sino for a peak
x = {}; jj=0
#sum20L
x[jj] = pd.DataFrame([[29, 'sum20L']], columns=['angle','peak']); jj = jj+1
x[jj] = pd.DataFrame([[209, 'sum20L']], columns=['angle','peak']); jj = jj+1
#sum21L
x[jj] = pd.DataFrame([[51, 'sum21L']], columns=['angle','peak']); jj = jj+1
x[jj] = pd.DataFrame([[189.5, 'sum21L']], columns=['angle','peak']); jj = jj+1
x[jj] = pd.DataFrame([[51+180, 'sum21L']], columns=['angle','peak']); jj = jj+1
x[jj] = pd.DataFrame([[10, 'sum21L']], columns=['angle','peak']); jj = jj+1
#sum11L
x[jj] = pd.DataFrame([[58, 'sum11L']], columns=['angle','peak']); jj = jj+1
x[jj] = pd.DataFrame([[164.5, 'sum11L']], columns=['angle','peak']); jj = jj+1
x[jj] = pd.DataFrame([[58+180, 'sum11L']], columns=['angle','peak']); jj = jj+1
x[jj] = pd.DataFrame([[164+180, 'sum11L']], columns=['angle','peak']); jj = jj+1
#sum12L
x[jj] = pd.DataFrame([[79, 'sum12L']], columns=['angle','peak']); jj = jj+1
x[jj] = pd.DataFrame([[147, 'sum12L']], columns=['angle','peak']); jj = jj+1
x[jj] = pd.DataFrame([[79+180, 'sum12L']], columns=['angle','peak']); jj = jj+1
x[jj] = pd.DataFrame([[147+180, 'sum12L']], columns=['angle','peak']); jj = jj+1
#sum02L
#x[jj] = pd.DataFrame([[112, 'sum02L']], columns=['angle','peak']); jj = jj+1
list_peaks_angles_orig = pd.concat(x)
print(list_peaks_angles_orig.sort_values('angle'))
list_peaks_angles = list_peaks_angles_orig.copy()


## Different domains
domain_angle_offset = [-12, -10, -4, -0.5, 0, 0.5, 1, 1.5, 2, 6, 8, 12, 13, 19, 24]
plt.figure(101, figsize=[20, 10]); plt.clf()
for ii, offset in enumerate(domain_angle_offset):  
    print(offset)
    angles_old = list_peaks_angles_orig['angle']
    angles_new = angles_old + offset
    list_peaks_angles['angle'] = angles_new

    ## Get sino
    width = 0
    sino_dm = get_combined_sino(sino_dict, list_peaks_angles.sort_values('angle'), width=width, verbose=1)
    ## Plot sino
    title_st = '{}\n offset={} and width={}'.format(filename, offset, width)
    plt.subplot(3,len(domain_angle_offset),ii+1)
    plot_sino(sino_dm, theta = sino_dict_dm['theta'], axis_x = sino_dict_dm['axis_x'], title_st=title_st, vlog10=[-0.1, 0.1], fignum=-1)
    #plot_angles(list_peaks_angles['angle'], fignum=51)    
    
    # Tomo recon
    plt.subplot(3,len(domain_angle_offset),len(domain_angle_offset)+ii+1)
    title_st = 'offset={}$^\circ$\nwidth={}'.format(offset, width)
    recon_all = get_plot_recon(sino_dm, theta = sino_dict_dm['theta'], rot_center=32, algorithms = ['fbp'], title_st=title_st, fignum=-1, colorbar=True)
    
    # Another width
    width = 1
    sino_dm = get_combined_sino(sino_dict, list_peaks_angles.sort_values('angle'), width=width, verbose=1)
    plt.subplot(3,len(domain_angle_offset),len(domain_angle_offset)*2+ii+1)
    title_st = 'offset={}$^\circ$\nwidth={}'.format(offset, width)
    recon_all = get_plot_recon(sino_dm, theta = sino_dict_dm['theta'], rot_center=32, algorithms = ['fbp'], title_st=title_st, fignum=-1, colorbar=True)




