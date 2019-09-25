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
infiles = glob.glob(os.path.join(source_dir, '*tomo_real_9*.tiff'))
#for ii in [2,3]: infiles.extend(glob.glob(os.path.join(source_dir, '*tomo_real_*00{}*.tiff'.format(ii))))
filename = infiles[0][0:infiles[0].find('real_')+5]
N_files = len(infiles)
print('N_files = {}'.format(N_files))
# e.g. ../raw/C8BTBT_0.1Cmin_tomo_real_9_x-3.600_th0.090_1.00s_2526493_000656_waxs.tiff'


########################################## 
# Load all data and plot sum
##########################################
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
# Load and plot to define roi
if True:
    temp2 = np.load(fn_out)
    plt.figure(100, figsize=[15,15]); plt.clf(); plt.title(fn_out)
    plt.imshow(np.log10(temp2), vmin=0.6, vmax=1.2); plt.colorbar()    
    get_peaks(infiles[0], verbose=1)

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
t0 = time.time()
with Parallel(n_jobs=4) as parallel:
    results = parallel( delayed(get_peaks)(infile) for infile in infiles )
print("\nLoad data and define peak roi: {:.0f} s".format(time.time()-t0))


# Pass to pd
df_peaks = pd.DataFrame()
for ii, df in enumerate(results):
    df_peaks = df_peaks.append(df, ignore_index=True)
print(df_peaks)


# Save 
df_peaks.to_csv('df_peaks_002_11L_1_1L')
 
##########################################
# Plot Sino
##########################################   
data_sort = df_peaks.sort_values(by=['pos_phi', 'pos_x'])
#data_sort_drop = data_002_sort[data_sort.pos_phi >=0]

theta = data_sort['pos_phi']
theta = theta.drop_duplicates()
theta = np.asarray(theta)

axis_x = data_sort['pos_x']
axis_x = axis_x.drop_duplicates()
axis_x = np.asarray(axis_x)
    
# Create projection from pd data
peak = 'sum11L'
proj_orig = data_sort[peak] # + data_sort['sum11L'] 
proj = proj_orig.values
proj = np.reshape(proj, (len(theta), 1, int(len(proj)/len(theta))) )
#proj = pow(proj,1.2)
proj[proj<15000] = 0
#proj[190:500,:,:] = 0

# Plot
plt.figure(31); plt.clf()
plt.subplot(121)
plt.imshow(proj[:,0,:], cmap='jet', aspect='auto')
plt.colorbar()
plt.title('{}\n{}'.format(filename, peak))
plt.xlabel('pos_x')
plt.ylabel('pos_phi')

plt.subplot(122)
plt.imshow(np.log10(proj[:,0,:]), cmap='jet', aspect='auto', extent = [axis_x[0], axis_x[-1], theta[-1], theta[0]])
plt.colorbar()
plt.title('log10('+peak+')')
  

##########################################
# Find center
##########################################   
#proj = tomopy.minus_log(proj)
flat = np.ones((1,1,proj.shape[2]))
dark = np.zeros((1,1,proj.shape[2]))
proj = tomopy.normalize(proj, flat, dark) # (proj-dark)/(flat-dark)

#rot_center = tomopy.find_center(proj, theta, init=290, ind=0, tol=0.5)
rot_center = tomopy.find_center(proj, theta, init=30, ind=0, tol=0.1)
print('Rotational center: {}'.format(rot_center))
if (rot_center>30) or (rot_center<20):
    rot_center = 25
    

##########################################
# Tomo reconstruction
##########################################      
# Keyword "algorithm" must be one of ['art', 'bart', 'fbp', 'gridrec', 'mlem', 'osem', 'ospml_hybrid', 
# 'ospml_quad', 'pml_hybrid', 'pml_quad', 'sirt', 'tv', 'grad'], or a Python method.

#recon = tomopy.recon(proj, theta, center=rot_center, algorithm=tomopy.lprec, lpmethod='tv', ncore=1, num_iter=512, reg_par=5e-4)

#rot_center = 29
algorithms = ['fbp', 'mlem', 'tv']
plt.figure(50, figsize=(15,5)); plt.clf()
for ii, algo in enumerate(algorithms):
    recon = tomopy.recon(proj, theta, center=rot_center, algorithm=algo)
    recon = tomopy.circ_mask(recon, axis=0, ratio=0.95)
    plt.subplot(1,3,ii+1)
    plt.imshow(recon[0, :,:], cmap='jet') #, vmin=0, vmax=1.5e7)
    plt.title('{}\n{}, center {}'.format(peak, algo, rot_center))
    plt.colorbar(); plt.show();

    
    
    
    
    
    
    
    
    
    
    
    
    