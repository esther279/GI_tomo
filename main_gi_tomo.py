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
out_dir = '../results_tomo/'
infiles = glob.glob(os.path.join(source_dir, '*C8BTBT_0.1Cmin_tomo_*.tiff'))
N_files = len(infiles); print('N_files = {}'.format(N_files))
#for ii in [2,3]: infiles.extend(glob.glob(os.path.join(source_dir, '*tomo_real_*00{}*.tiff'.format(ii))))
# e.g. ../raw/C8BTBT_0.1Cmin_tomo_real_9_x-3.600_th0.090_1.00s_2526493_000656_waxs.tiff'

flag_load_raw_data = 0
flag_get_peaks = 0;  flag_LinearSubBKG = 1
flag_load_peaks = 1
flag_tomo = 1

filename = infiles[0][infiles[0].find('C8BTBT'):infiles[0].find('tomo_')+5]
#filename = 'C8BTBT_0.1Cmin_tomo'
if os.path.exists(out_dir) is False: os.mkdir(out_dir)

### Define peak roi from scattering pattern
peak_list = [
        # center, size, peak
        [[575, 479], [60, 10], 'sum002'],
        # 11L
        [[525, 735], [180, 10], 'sum11L'],
        [[525, 223], [180, 10], 'sum11Lb'],
        # 02L
        [[603, 787-3], [30, 10], 'sum02L'],
        [[603, 172], [30, 10], 'sum02Lb'],
        # 12L
        [[589, 848], [58, 6], 'sum12L'], 
        [[589, 110], [58, 6], 'sum12Lb'],
        # 20L
        [[323-6, 903], [60, 15], 'sum20L'],
        [[323, 56], [30, 15], 'sum20Lb'],
        # 21L
        [[280, 936], [40, 15], 'sum21L'],
        [[280, 26], [40, 15], 'sum21Lb'],
        # Si
        [[400, 809], [12, 12], 'sumSi'],
        [[400, 151], [12, 12], 'sumSib'],
        # background
        [[560, 440], [30,30], 'sumBKG0'],
        ]
fn_out = out_dir+'peak_list'
fn_out = check_file_exist(fn_out)
np.save(fn_out, peak_list)


# =============================================================================
# Load all/some data and plot sum
# =============================================================================
if flag_load_raw_data:
    t0 = time.time()   
    fraction = 10  # Quick checck peak positions
    for ii, infile in enumerate(infiles):
        if ii%fraction==0: 
            print("{}/{}, {}".format(ii, N_files, infile))
            temp = Image.open(infile).convert('I')
            data = np.copy(np.asarray(temp))
            if ii==0:
                data_sum = data
            else:
                data_sum = data_sum+data
    data_avg = data_sum/np.size(infiles)*fraction    
    print("Data loading: {:.0f} s".format(time.time()-t0))
        
    # Plot
    plt.figure(1); plt.clf()
    plt.imshow(np.log10(data_avg), vmin=0.6, vmax=1.5)
    plt.colorbar()
    plt.title('Average over {} data (fraction=1/{}) \n {}'.format(N_files,fraction,infiles[0]))
    fn_out = out_dir+filename+'_avg'
    fn_out = check_file_exist(fn_out)
    plt.savefig(fn_out, format='png')

    # Save as npy
    fn_out = out_dir+'data_avg'
    fn_out = check_file_exist(fn_out)
    np.save(fn_out, data_avg)
    if False:
        data_avg = np.load(fn_out+'.npy')
    
    #### Plot to define roi
    plt.figure(100, figsize=[12,12]); plt.clf(); plt.title(fn_out)
    plt.imshow(np.log10(data_avg), vmin=0.3, vmax=1.5); plt.colorbar()    
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
            results = parallel( delayed(get_peaks)(infile, peak_list, verbose=1, flag_LinearSubBKG=flag_LinearSubBKG) for infile in infiles )
    else:
        results = []
        for ii, infile in enumerate(infiles):
            #if ii%10==0:
            temp = get_peaks(infile, peak_list, verbose=1, flag_LinearSubBKG=flag_LinearSubBKG)
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
 
    # Calculate area
    areas = calc_area_peakROI(peak_list)
    
# =============================================================================
# Get sino for each peak
# =============================================================================
if flag_load_peaks:
    df_peaks = pd.read_csv(out_dir+'df_peaks_all_subbg{}'.format(flag_LinearSubBKG))

## Create sino from pd data
list_peaks = []
data_sort, sino_dict = get_sino_from_data(df_peaks, list_peaks=list_peaks, flag_rm_expbg=1, flag_thr=1) #flag_thr=2 for binary
print(sino_dict['list_peaks'])
sino_sum = get_sino_sum(sino_dict)
sino_dict['areas'] = calc_area_peakROI(peak_list) #assuming list_peaks are the same as peak_list

## Plot sino
plot_sino(sino_dict, fignum=30, title_st=filename, vlog10=[0, 5.5])

fn_out = out_dir+filename+'peaks_sino' 
fn_out = check_file_exist(fn_out)
plt.savefig(fn_out, format='png')
    
## Do and plot recon
if flag_tomo:
    recon_all = get_plot_recon(sino_dict, rot_center=32, algorithms = ['gridrec', 'fbp', 'tv'], fignum=40)

    fn_out = out_dir+filename+'peaks_sino_tomo_subbg'+str(flag_LinearSubBKG); 
    fn_out = check_file_exist(fn_out)
    plt.savefig(fn_out, format='png')

   
# =============================================================================
# Label peak positons (deg) a sino
# =============================================================================
list_peaks = sino_dict['list_peaks']
flag_log10 = 0 # Use 1 only for plotting
flag_save_png = 1

x = {}; jj=0
N = len(list_peaks[0:-1])
plt.figure(10, figsize=[15, 8]); plt.clf()
for ii, peak in enumerate(list_peaks[0:-1]):
#peak =  'sum20L'
#if 1:
    sino, sum_sino, theta = get_sino_from_a_peak(sino_dict, peak) # which peak roi
    if flag_log10: 
        sum_sino = np.log10(sum_sino)
    plt.subplot(N,1,ii+1)
    plt.plot(theta, sum_sino);  
    plt.axis('off')     
    plt.legend([peak], loc='upper left')
    peaks_idx = label_peaks(theta, sum_sino, onedomain=1)
    
    # Store peaks and corresponding angles to a df for reconstructing a domain
    for angle in theta[peaks_idx]:
        if angle<181:
            x[jj] = pd.DataFrame([[angle, peak]], columns=['angle','peak'])
            jj = jj+1
    
# Save to png
if flag_save_png:
    fn_out = out_dir+'peak_deg' #+peak
    fn_out = check_file_exist(fn_out)
    plt.savefig(fn_out, format='png')

# =============================================================================
# Check angles      
# ============================================================================= 
temp_list = pd.concat(x)
print(temp_list) #print(list_peaks_angles_orig.sort_values('angle'))

## Remove peaks not needed for sino
print('## Compare the list with the figure and drop unwanted peaks.')
list_peaks_angles_orig = temp_list[temp_list.peak !='sumSi']
list_peaks_angles_orig = list_peaks_angles_orig[list_peaks_angles_orig.peak !='sumSib']
list_peaks_angles_orig = list_peaks_angles_orig.drop([24])   #list_peaks_angles_orig.copy()
list_peaks_angles_orig = list_peaks_angles_orig.drop([29]) 
print(list_peaks_angles_orig)
plot_angles(list_peaks_angles_orig['angle'], fignum=45)    


# =============================================================================
# Different domains
# =============================================================================
plt.figure(200, figsize=[20, 10]); plt.clf()
sino, sum_sino, theta = get_sino_from_a_peak(sino_dict, 'sum11L') #choose
plt.plot(theta, sum_sino);  
peaks_idx = label_peaks(theta, sum_sino, onedomain=0)
print(*theta[peaks_idx], sep=', ')

print('## Select the main peaks for reconstruction of different domains. See above for recommendations.')
#domain_angle_offset = np.asarray([197.5, 201.0, 205.0, 209.0, 215.0, 221.0, 233.0]) - 209.0
domain_angle_offset = np.asarray([154.0, 157.5, 160.5, 161.5, 165.5, 170.5, 171.5, 188.5, 189.5,]) - 165.5
domain_angle_offset = np.append(domain_angle_offset, 12)
#domain_angle_offset = np.append(domain_angle_offset, np.arange(-2,2.5,0.5))
domain_angle_offset = np.sort(domain_angle_offset)
print('domain_angle_offset = {}'.format(domain_angle_offset))

## Do recon for each domain
recon_all_list = []
sino_all_list = []
plt.figure(203, figsize=[20, 10]); plt.clf()
list_peaks_angles = list_peaks_angles_orig.copy()

for ii, offset in enumerate(domain_angle_offset):  
    print(offset)
    angles_old = list_peaks_angles_orig['angle']
    angles_new = angles_old + offset
    list_peaks_angles['angle'] = angles_new

    ## Get sino
    flag_normal = 3 # 1(normalize max to 1), 2(divided by the ROI area), 3 (binary)
    width = 2
    sino_dm = get_combined_sino(sino_dict, list_peaks_angles.sort_values('angle'), width=width, flag_normal=flag_normal, verbose=1)
    ## Plot sino
    title_st = '{}\nflag_normal={}'.format(filename, flag_normal) if ii==0 else ''
    plt.subplot(2,len(domain_angle_offset),ii+1)
    plot_sino((sino_dm), theta = sino_dict['theta'], axis_x = sino_dict['axis_x'], title_st=title_st, fignum=-1)
    #plot_angles(list_peaks_angles['angle'], fignum=51)    
    
    ## Tomo recon
    plt.subplot(2,len(domain_angle_offset),len(domain_angle_offset)+ii+1)
    title_st = '[{}] ori={}$^\circ$'.format(ii,offset)
    temp = get_plot_recon(sino_dm, theta = sino_dict['theta'], rot_center=30, algorithms = ['fbp'], title_st=title_st, fignum=-1, colorbar=True)
    sino_all_list.append(sino_dm)
    recon_all_list.append(np.squeeze(temp['_fbp']))
        
fn_out = out_dir+'recon'
fn_out = check_file_exist(fn_out)
plt.savefig(fn_out, format='png')


# =============================================================================
# Overlap three domains spatially
# =============================================================================
domains_use = [0, 3, 4]   #overlay these domains

plt.figure(400, figsize=[20,10]); plt.clf()
rgb = 'RGB'
channel=0; overlay = []
for ii in domains_use:      
    recon = recon_all_list[ii]
    thr = np.max(recon)*0.7
    recon_binary = recon.copy()
    recon_binary[recon<thr] = 0
    recon_binary = recon_binary/np.max(recon)
    #recon_binary[recon>=thr] = 1
    ax = plt.subplot2grid((3, 7), (channel, 0), colspan=2); 
    image_channel = np.asarray(image_RGB(recon_binary, rgb[channel]))
    if overlay==[]:
        overlay = image_channel
    else: 
        overlay += image_channel
    plt.imshow(image_channel); plt.axis('off')
    plt.title('ori = {:.1f}$^\circ$'.format(domain_angle_offset[ii]))
    channel += 1
ax = plt.subplot2grid((3, 7), (0, 2), rowspan=3, colspan=4); ax.cla()
ax.set_facecolor('k')    
plt.imshow(overlay)  #, origin='lower')    
   
## Save to png
if flag_save_png:
    fn_out = out_dir+'recon_overlay{}{}{}'.format(overlay_rgb[0], overlay_rgb[1], overlay_rgb[2])
    fn_out = check_file_exist(fn_out)
    plt.savefig(fn_out, format='png')


# =============================================================================
# Plot all recons after threshold
# =============================================================================
recon_merged = np.zeros([recon_all_list[0].shape[0], recon_all_list[0].shape[1]])
Ndomain = len(domain_angle_offset)
plt.figure(300, figsize=[20,10]); plt.clf()
for ii, recon in enumerate(recon_all_list):
    thr = np.max(recon)*0.5
    print(thr)
    recon_binary = recon.copy()
    recon_binary[recon<thr] = np.nan
    recon_binary[recon>=thr] = domain_angle_offset[ii]
    recon_merged = recon_merged + recon_binary
    plt.subplot(1,Ndomain,ii+1)  
    plt.imshow(recon_binary); plt.axis('off')
    plt.title('{}\nori = {:.1f}$^\circ$'.format(ii,domain_angle_offset[ii]))



# =============================================================================
# Generate a guess 
# =============================================================================
domains_use = [0, 3, 4, 6]  #np.arange(0, len(recon_all_list))
recon_all_list_normal = []
for ii in domains_use:      
    recon = recon_all_list[ii]
    recon_all_list_normal.append(recon/np.max(recon))
mask = (recon!=0).astype(float)
mask_nan = mask.copy()
mask_nan[mask==0] = np.nan

temp_angle = domain_angle_offset[domains_use]
domains_recon = mask_nan*temp_angle[np.argmax(recon_all_list_normal,0)]

plt.figure(22); plt.clf()
plt.imshow(domains_recon, cmap='summer')
plt.colorbar()
plt.title('orientation angles {}'.format(temp_angle))


## Save to npy
if 1:    
    fn_out = out_dir+'domains_recon'
    fn_out = check_file_exist(fn_out)
    plt.savefig(fn_out, format='png')
    
    fn_out = out_dir+'domains_recon.npy'
    np.save(fn_out, domains_recon)
    
    fn_out = out_dir+'sino_all_list.npy'
    np.save(fn_out, sino_all_list)
    
    rot_angles = np.asarray(list_peaks_angles_orig.sort_values('angle')['angle'])
    fn_out = out_dir+'rot_angles.npy'
    np.save(fn_out, rot_angles)







