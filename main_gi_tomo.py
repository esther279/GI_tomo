#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, glob, time, sys
HOME_PATH = '/home/etsai/BNL/Research/GIWAXS_tomo_2020C1/RLi6/'
GI_TOMO_PATH = HOME_PATH+'GI_tomo/'
GI_TOMO_PATH in sys.path or sys.path.append(GI_TOMO_PATH)

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import tomopy
from joblib import Parallel, delayed
from scipy import signal

import analysis.peaks as peaks
import analysis.tomo as tomo
import analysis.seg as seg
import analysis.util as util

# =============================================================================
# Specify input
# =============================================================================
os.chdir(HOME_PATH)
source_dir = './waxs/raw/'
out_dir = './results_tomo/'
infiles = glob.glob(os.path.join(source_dir, '*C8BTBT_0.1Cmin_tomo_*.tiff'))
N_files = len(infiles); print('N_files = {}'.format(N_files))

flag_load_raw_data = 0
flag_get_peaks = 0;  flag_LinearSubBKG = 1
flag_load_peaks = 1
flag_tomo = 1

filename = infiles[0][infiles[0].find('C8BTBT'):infiles[0].find('tomo_')+4]
print(filename)
if os.path.exists(out_dir) is False: os.mkdir(out_dir)


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
    plt.imshow(np.log10(data_avg), vmin=1.1, vmax=1.8)
    plt.colorbar()
    plt.title('Average over {} data (fraction=1/{}) \n {}'.format(N_files,fraction,infiles[0]))
    
    # Save png
    fn_out = out_dir+filename+'_avg.png'
    fn_out = util.check_file_exist(fn_out)
    plt.savefig(fn_out, format='png')

    # Save as npy
    fn_out = out_dir+'data_avg'
    fn_out = util.check_file_exist(fn_out)
    np.save(fn_out, data_avg)
    if False:
        fn_out = out_dir+'data_avg'
        data_avg = np.load(fn_out+'.npy')
    
    # Save as tiff
    if True:
        final_img = Image.fromarray((data_avg).astype(np.uint32))
        infile_done = out_dir+filename+'_data_avg.tiff'
        final_img.save(infile_done)     
        
        # Plot qr (after use SciAnalysis on the tiff file)
        if True:
            fn = './waxs/analysis/qr_image/TOMO_T1_real_data_avg.npz'
            qinfo = np.load(fn)
            qr_image = qinfo['image']
            x_axis = qinfo['x_axis']
            y_axis = qinfo['y_axis']
            extent = (np.nanmin(x_axis), np.nanmax(x_axis), np.nanmin(y_axis), np.nanmax(y_axis))
            plt.figure(11, figsize=[12,8]); plt.clf()
            plt.imshow(np.log10(qr_image), origin='bottom', extent=extent, vmin=1.1, vmax=1.8) 
            plt.ylim(0, np.nanmax(y_axis))
            plt.grid(axis='x'); plt.colorbar()
            plt.title(fn)
            
            fn_out = out_dir+filename+'_qr.png'
            plt.savefig(fn_out, format='png')
        
        # Load and plot
        if False:
            temp = Image.open(infile_done).convert('I') # 'I' : 32-bit integer pixels
            data_avg_infile = np.copy( np.asarray(temp) )
            plt.figure(2); plt.clf()
            plt.imshow(data_avg_infile)
            plt.clim(0, 20)
            plt.colorbar()
            plt.show()
    
    ### Define peak roi from scattering pattern
    peak_list = [
            # center, size, peak
            [[575, 471], [60, 10], 'sum002'],
            # 01L
            [[445, 577], [40, 10], 'sum01L'],
            [[445, 366], [40, 10], 'sum01Lb'],
            # 11L
            [[520, 654], [200, 20], 'sum11L'],
            [[520, 291], [200, 20], 'sum11Lb'],
            # 02L
            [[605, 688], [100, 10], 'sum02L'],
            [[574, 255], [30, 10], 'sum02Lb'],
            [[189, 679], [10, 10], 'sum02Lx'],
            [[189, 262], [10, 10], 'sum02Lbx'],
            # 12L
            [[540, 737], [230, 15], 'sum12L'], 
            [[520, 206], [200, 15], 'sum12Lb'],
            # 20L
            [[540, 770], [230, 16], 'sum20L'],
            [[520, 173], [200, 16], 'sum20Lb'],
            # 21L
            [[520, 794], [200, 15], 'sum21L'],
            [[520, 151], [200, 15], 'sum21Lb'],
            # 03L
            [[520, 813], [200, 15], 'sum03L'],
            [[520, 129], [200, 15], 'sum03Lb'],
            # 13L
            [[583, 844], [70, 15], 'sum13L'],
            [[583, 101], [70, 15], 'sum13Lb'],
            # 22L
            [[462, 857], [10, 20], 'sum22L'],
            [[462, 88], [10, 20], 'sum22Lb'],
            # 
            [[595, 939], [10, 10],  [390, 949], [10, 10], 'sum23L'],
            [[480, 960], [50, 15], 'sum31L'],
            #[[465, 969], [10, 10], 'sum31L'],
            # Si
            #[[400, 809], [12, 12], 'sumSi'],
            #[[400, 151], [12, 12], 'sumSib'],
            # background
            [[570, 430], [30,30], 'sumBKG0'],
            [[385, 430], [30,30], 'sumBKG1'],
            ]
    #### Plot to define roi
    fig = plt.figure(100, figsize=[12,12]); plt.clf(); plt.title(filename+'\n'+fn_out)
    ax = fig.add_subplot(111)
    ax.imshow(np.log10(data_avg), vmin=1.1, vmax=1.8)
    peaks.get_peaks(infiles[0], peak_list, phi_max=180, verbose=2)
    
    ## Save png
    fn_out = out_dir+filename+'_peak_roi.png'
    fn_out = util.check_file_exist(fn_out)
    plt.savefig(fn_out, format='png')
    
    # Save peak_list in npy
    fn_out = out_dir+'peak_list'
    fn_out = util.check_file_exist(fn_out)
    np.save(fn_out, peak_list)

        
# =============================================================================
# Get peaks from raw tiff files
# =============================================================================
if flag_get_peaks:   
    
    t0 = time.time()
    flag_load_parellel = 0  # Sometimes parallel doesn't work..
    if flag_load_parellel:
        with Parallel(n_jobs=3) as parallel:
            results = parallel( delayed(peaks.get_peaks)(infile, peak_list, phi_max=180, verbose=1, flag_LinearSubBKG=flag_LinearSubBKG) for infile in infiles )
    else:
        results = []
        for ii, infile in enumerate(infiles):
            #if ii%10==0:
            temp = peaks.get_peaks(infile, peak_list, phi_max=180, verbose=1, flag_LinearSubBKG=flag_LinearSubBKG)
            results.append(temp)
    print("\nLoad data and define peak roi: {:.0f} s".format(time.time()-t0))
    
    
    # Pass to pd
    df_peaks = pd.DataFrame()
    for ii, df in enumerate(results):
        df_peaks = df_peaks.append(df, ignore_index=True)
    print(df_peaks)
    print(df_peaks.columns)
    
    # Save 
    fn_out = out_dir+'df_peaks_all_subbg{}'.format(flag_LinearSubBKG)
    fn_out = util.check_file_exist(fn_out)
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
data_sort, sino_dict = tomo.get_sino_from_data(df_peaks, list_peaks=list_peaks, flag_rm_expbg=1, thr=0.5, binary=None) 
print(sino_dict['list_peaks'])
sino_sum = tomo.get_sino_sum(sino_dict)
sino_dict['areas'] = peaks.calc_area_peakROI(peak_list) #assuming list_peaks are the same as peak_list

## Plot sino
tomo.plot_sino(sino_dict, fignum=30, title_st=filename, vlog10=[0, 5.5])

fn_out = out_dir+filename+'peaks_sino' 
fn_out = util.check_file_exist(fn_out)
plt.savefig(fn_out, format='png')
    
## Do and plot recon
if flag_tomo:
    recon_all = tomo.get_plot_recon(sino_dict, rot_center=32, algorithms = ['gridrec', 'fbp', 'tv'], fignum=40)

    fn_out = out_dir+filename+'peaks_sino_tomo_subbg'+str(flag_LinearSubBKG); 
    fn_out = tomo.check_file_exist(fn_out)
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
    sino, sum_sino, theta = tomo.get_sino_from_a_peak(sino_dict, peak) # which peak roi
    if flag_log10: 
        sum_sino = np.log10(sum_sino)
    plt.subplot(N,1,ii+1)
    plt.plot(theta, sum_sino);  
    plt.axis('off')     
    plt.legend([peak], loc='upper left')
    peaks_idx = peaks.label_peaks(theta, sum_sino, onedomain=1)
    
    # Store peaks and corresponding angles to a df for reconstructing a domain
    for angle in theta[peaks_idx]:
        if angle<181:
            x[jj] = pd.DataFrame([[angle, peak]], columns=['angle','peak'])
            jj = jj+1
    
# Save to png
if flag_save_png:
    fn_out = out_dir+'peak_deg' #+peak
    fn_out = util.check_file_exist(fn_out)
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
tomo.plot_angles(list_peaks_angles_orig['angle'], fignum=45)    


# =============================================================================
# Different domains
# =============================================================================
plt.figure(200, figsize=[20, 10]); plt.clf()
sino, sum_sino, theta = tomo.get_sino_from_a_peak(sino_dict, 'sum11L') #choose
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
list_peaks_angles = list_peaks_angles_orig.copy()

plt.figure(203, figsize=[20, 10]); plt.clf()
for ii, offset in enumerate(domain_angle_offset):  
    print(offset)
    angles_old = list_peaks_angles_orig['angle']
    angles_new = angles_old + offset
    list_peaks_angles['angle'] = angles_new

    ## Get sino
    flag_normal = 3 # 1(normalize max to 1), 2(divided by the ROI area), 3 (binary)
    width = 2
    sino_dm = tomo.get_combined_sino(sino_dict, list_peaks_angles.sort_values('angle'), width=width, flag_normal=flag_normal, verbose=1)
    ## Plot sino
    title_st = '{}\nflag_normal={}'.format(filename, flag_normal) if ii==0 else ''
    plt.subplot(2,len(domain_angle_offset),ii+1)
    tomo.plot_sino((sino_dm), theta = sino_dict['theta'], axis_x = sino_dict['axis_x'], title_st=title_st, fignum=-1)
    #plot_angles(list_peaks_angles['angle'], fignum=51)    
    
    ## Tomo recon
    plt.subplot(2,len(domain_angle_offset),len(domain_angle_offset)+ii+1)
    title_st = '[{}] ori={}$^\circ$'.format(ii,offset)
    temp = tomo.get_plot_recon(sino_dm, theta = sino_dict['theta'], rot_center=30, algorithms = ['fbp'], title_st=title_st, fignum=-1, colorbar=True)
    sino_all_list.append(sino_dm)
    recon_all_list.append(np.squeeze(temp['_fbp']))
        
fn_out = out_dir+'recon'
fn_out = util.check_file_exist(fn_out)
plt.savefig(fn_out, format='png')


# =============================================================================
# Overlap three domains spatially
# =============================================================================
domains_use = [0, 1, 3, 4, 6, 7, 8]   #overlay these domains
rgb = 'RGBWCMY'
channel=0; overlay = []

plt.figure(400, figsize=[20,10]); plt.clf()
for ii in domains_use:      
    recon = recon_all_list[ii]
    
    if 1: ## Threshold
        if ii==4:
            thr = 0.55
        else:
            thr = 0.55
        recon_plot = seg.do_thr(recon, thr)
        
    else: ## Segmentation
        center = np.unravel_index(np.argmax(recon, axis=None), recon.shape)
        center = np.flip(np.asarray(center))
        print(center)
        if ii==0:
            centers = [center, [15, 20]]
        elif ii==4: 
            centers = [center, [18,27]]
        else:
            centers = [center]
        recon_plot = seg.do_segmentation(recon, centers, width=2, fignum=0)
    
    ## Plot
    ax = plt.subplot2grid((7, 7), (channel, 0), colspan=2); 
    image_channel = np.asarray(util.image_RGB(recon_plot, rgb[channel]))
    if overlay==[]:
        overlay = image_channel
    else: 
        overlay += image_channel
    plt.imshow(image_channel); plt.axis('off')
    plt.title('ori = {:.1f}$^\circ$'.format(domain_angle_offset[ii]))
    channel += 1
    
ax = plt.subplot2grid((7, 7), (0, 2), rowspan=3, colspan=4); ax.cla()
ax.set_facecolor('k')    
plt.imshow(overlay)  #, origin='lower')    
plt.title('thr = {}'.format(thr))
   

## Save to png
if flag_save_png:
    fn_out = out_dir+'recon_overlay{}{}{}'.format(domains_use[0], domains_use[1], domains_use[2])
    fn_out = check_file_exist(fn_out)
    plt.savefig(fn_out, format='png')


# =============================================================================
# Plot all recons after threshold
# =============================================================================
recon_merged = np.zeros([recon_all_list[0].shape[0], recon_all_list[0].shape[1]])
Ndomain = len(domain_angle_offset)

plt.figure(300, figsize=[20,10]); plt.clf()
for ii, recon in enumerate(recon_all_list):
    thr = np.max(recon)*0.55
    print(thr)
    recon_binary = recon.copy()
    recon_binary[recon<thr] = -20 #np.nan
    recon_binary[recon>=thr] = domain_angle_offset[ii]
    recon_merged = recon_merged + recon_binary
    
    plt.subplot(1,Ndomain+1,ii+1)  
    plt.imshow(recon_binary); plt.axis('off')
    plt.title('{}\nori = {:.1f}$^\circ$'.format(ii,domain_angle_offset[ii]))
plt.subplot(1,Ndomain+1,Ndomain+1)  
plt.imshow(recon_merged)


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
    fn_out = util.check_file_exist(fn_out)
    plt.savefig(fn_out, format='png')
    
    fn_out = out_dir+'domains_recon.npy'
    np.save(fn_out, domains_recon)
    
    fn_out = out_dir+'sino_all_list.npy'
    np.save(fn_out, sino_all_list)
    
    rot_angles = np.asarray(list_peaks_angles_orig.sort_values('angle')['angle'])
    fn_out = out_dir+'rot_angles.npy'
    np.save(fn_out, rot_angles)







