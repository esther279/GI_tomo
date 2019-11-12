#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, glob, time, sys
import numpy as np
import matplotlib.pyplot as plt
import copy
import tomopy
from scipy.signal import find_peaks

# =============================================================================
# Load dataframe into a dictionary and do preprocessing
# =============================================================================
def get_sino_from_data(data, list_peaks=[], flag_rm_expbg=1, flag_thr=0, flag_align=1):
    
    if list_peaks==[]:
        aa = [1 if 'sum' in temp else 0 for temp in data.keys()]
        idx = aa.index(max(aa))
        list_peaks = list(data.columns[idx:])
        
    data_sort = data.sort_values(by=['pos_phi', 'pos_x'])
    
    theta = data_sort['pos_phi']
    theta = theta.drop_duplicates()
    theta = np.asarray(theta)
    
    axis_x = data_sort['pos_x']
    axis_x = axis_x.drop_duplicates()
    axis_x = np.asarray(axis_x)

    if flag_rm_expbg and'sumBKG0' in data_sort.keys():
        temp = copy.deepcopy(data_sort['sumBKG0'])
        proj_bkg = temp.values
        proj_bkg = np.reshape(proj_bkg, (len(theta), len(axis_x)) )  
        proj_bkg = proj_bkg/np.max(proj_bkg)
        bkg_max_idx = np.where(proj_bkg==1)
            
    sino_allpeaks = np.zeros([len(theta),  len(axis_x), len(list_peaks)])
    for ii, peak in enumerate(list_peaks):
        proj = copy.deepcopy(data_sort[peak]).values #- data_sort['sumBKG0'].values*4
        proj = np.reshape(proj, (len(theta), len(axis_x)) )
        
        if flag_rm_expbg and'sumBKG0' in data_sort.keys():
            #print(proj[bkg_max_idx])
            proj = proj - proj_bkg*proj[bkg_max_idx]            
        #proj = proj[:,:,6:]
        #proj = pow(proj,1.2)    
        
        if flag_thr!=0:
            thr = np.median(proj)*7
            print('thr = {}'.format(thr))
            proj[proj<thr] = 0
            #proj[proj>=thr] = 1
            
        if flag_align:                
            ## Manually align sino
            old = proj[:,31]
            proj[:,31] = np.roll(old,-1)   
            
        
        sino_allpeaks[:,:,ii] = proj
        
    sino_dict = {}
    sino_dict['sino_allpeaks'] = sino_allpeaks
    sino_dict['theta'] = theta
    sino_dict['axis_x'] = axis_x
    sino_dict['list_peaks'] = list_peaks
    
    return data_sort, sino_dict


# =============================================================================
# Sum sino over all peaks
# =============================================================================
def get_sino_sum(sino_data):
    if type(sino_data)==dict:
        sino_allpeaks = sino_data['sino_allpeaks']
        theta = sino_data['theta']
        axis_x = sino_data['axis_x']
        list_peaks = sino_data['list_peaks']
    else:
        return sino_data
        
    sino_sum = np.zeros([sino_allpeaks.shape[0], sino_allpeaks.shape[1]])
    for ii in np.arange(0, sino_allpeaks.shape[2]):
        sino_sum = sino_sum + sino_allpeaks[:,:,ii]
        
    sino_data['sino_sum'] = sino_sum
    
    return sino_sum
        
# =============================================================================
# Plot sinogram
# =============================================================================
def plot_sino(sino_data, fignum=30, theta=[0, 1], axis_x=[0, 1], title_st='sino', vlog10=[0, 6]):   
    if type(sino_data)==dict:
        sino_allpeaks = sino_data['sino_allpeaks']
        theta = sino_data['theta']
        axis_x = sino_data['axis_x']
        list_peaks = sino_data['list_peaks']
    else:
        sino_data = np.asarray(sino_data)
        sino_allpeaks = np.reshape(sino_data, (sino_data.shape[0], sino_data.shape[1], 1))
        list_peaks = []
    
    if fignum>0: plt.figure(fignum, figsize=[12,12]); plt.clf()    
    Npeaks =  sino_allpeaks.shape[2]
    for ii in np.arange(0, sino_allpeaks.shape[2]):
        sino = sino_allpeaks[:,:,ii]
        peak = list_peaks[ii] if list_peaks!=[] else ''
        
        if fignum>0: 
            plt.subplot(2,Npeaks,ii+1)
        plt.imshow(sino, cmap='jet', aspect='auto') #, extent = [axis_x[0], axis_x[-1], theta[-1], theta[0]])
        plt.axis('off')
        if fignum>0:
            if ii==0: 
                plt.title('{}\n{}'.format(title_st, peak), fontweight='bold')
                plt.axis('on');
            elif ii%2: plt.title('{}'.format(peak), fontweight='bold')
            else: plt.title('{}'.format(peak))
            v1 = np.linspace(sino.min(), sino.max(), 2, endpoint=True)
            cb = plt.colorbar(orientation='horizontal', pad=0.05, ticks=v1)
            cb.ax.set_xticklabels(["{:.1e}".format(v) if v>0 else "" for v in v1])
        
        if fignum>0: 
            plt.subplot(2,Npeaks,Npeaks+ii+1)
            plt.imshow(np.log10(sino), cmap='jet', aspect='auto', extent = [axis_x[0], axis_x[-1], theta[-1], theta[0]], vmin=vlog10[0], vmax=vlog10[1])
            plt.axis('off')
            if ii==0: 
                plt.axis('on'); plt.title('log10')
                plt.xlabel('pos_x (mm)')
                plt.ylabel('pos_phi (deg)')    
            else: plt.axis('off')
            if ii==sino_allpeaks.shape[2]-1:
                plt.colorbar(orientation='horizontal', pad=0.01)    

# =============================================================================
# Do recon and plot
# =============================================================================
def get_plot_recon(sino_data, theta = [], rot_center=10, algorithms = ['art', 'gridrec', 'fbp'], title_st='recon', fignum=40, colorbar=False):
    if type(sino_data)==dict:
        sino_allpeaks = sino_data['sino_allpeaks']
        theta = sino_data['theta']
        axis_x = sino_data['axis_x']
        list_peaks = sino_data['list_peaks']
    else:
        sino_data = np.asarray(sino_data)
        sino_allpeaks = np.reshape(sino_data, (sino_data.shape[0], sino_data.shape[1], 1))
        list_peaks = []
    
    if fignum>0: plt.figure(fignum, figsize=[12,12]); plt.clf()    
    Npeaks =  sino_allpeaks.shape[2]  
    recon_all = {}
    for ii in np.arange(0, sino_allpeaks.shape[2]):
        sino = sino_allpeaks[:,:,ii]
        sino = np.reshape(sino, (sino.shape[0], 1, sino.shape[1]))
        peak = list_peaks[ii] if list_peaks!=[] else ''
        
        #proj = tomopy.minus_log(proj)
        flat = np.ones((1,1,sino.shape[2]))
        dark = np.zeros((1,1,sino.shape[2]))
        sino = tomopy.normalize(sino, flat, dark) # (sino-dark)/(flat-dark)
        
        ## Get rotational center if not specified
        cen_init = sino.shape[2]/2
        if rot_center<=0:
            rot_center = tomopy.find_center(sino, theta, init=cen_init, ind=0, tol=0.1)
            print('Rotational center: {}'.format(rot_center))
            if (rot_center>cen_init+10) or (rot_center<cen_init-10):
                rot_center = sino.shape[2]/2

        ## Tomo reconstruction 
        # Keyword "algorithm" must be one of ['art', 'bart', 'fbp', 'gridrec', 'mlem', 'osem', 'ospml_hybrid', 
        # 'ospml_quad', 'pml_hybrid', 'pml_quad', 'sirt', 'tv', 'grad'], or a Python method.
        #recon = tomopy.recon(proj, theta, center=rot_center, algorithm=tomopy.lprec, lpmethod='tv', ncore=1, num_iter=512, reg_par=5e-4)
        
        #rot_center = cen_init
        #plt.figure(50)
        for jj, algo in enumerate(algorithms):
            recon = tomopy.recon(sino, theta, center=rot_center, algorithm=algo)
            recon = tomopy.circ_mask(recon, axis=0, ratio=0.95)
            if fignum>0: plt.subplot(len(algorithms), Npeaks, (jj)*Npeaks+ii+1)
            plt.imshow(recon[0, :,:], cmap='jet') #, vmin=0, vmax=1.5e7)
            if title_st==[]:
                if ii%2: plt.title('{}\n{}, cen{:.1f}'.format(peak, algo, float(rot_center)), fontweight='bold')
                else: plt.title('{}\n{}, cen{:.1f}'.format(peak, algo, float(rot_center)))
            else:
                plt.title(title_st)

            recon_all[peak+'_'+algo] = recon
            if colorbar:
                v1 = np.linspace(recon.min(), recon.max(), 2, endpoint=True)
                cb = plt.colorbar(orientation='horizontal', pad=0.05, ticks=v1)
                cb.ax.set_xticklabels(["{:.1f}".format(v) if v>0 else "" for v in v1])
            plt.axis('off')
            
    return recon_all
    
# =============================================================================
# Combine data from different peaks into one sino for a domain
# =============================================================================
def get_combined_sino(sino_dict, list_peaks_angles, width=0, verbose=0):
    sino_allpeaks = sino_dict['sino_allpeaks']
    theta = sino_dict['theta']
    #list_peaks = sino_dict['list_peaks']
    
    peaks = np.asarray(list_peaks_angles['peak'])
    angles = np.asarray(list_peaks_angles['angle'])
    
    if verbose>0: print('------')
    sino_dm = np.zeros([sino_allpeaks.shape[0], sino_allpeaks.shape[1]])
    for ii in np.arange(0,len(list_peaks_angles)):
        sino = get_sino_from_a_peak(sino_dict, peaks[ii])  # get the sino for this peak (eg 'sum11L')
        angle = angles[ii]
        if verbose>0: print('angle = {}, peak = {}'.format(angle, peaks[ii]))
        
        angle_idx = get_idx_angle(theta, theta=angle)
        temp = get_proj_from_sino(sino,  angle_idx, width)  # get the projection at the angle        
        sino_dm[angle_idx-width:angle_idx+width+1, :] = temp

    sino_dict['sino_dm'] = sino_dm
    if verbose>0: print('------')
    
    return sino_dm


# =============================================================================
# Get the sino for a certain peak (eg 'sum11L')
# =============================================================================
def get_sino_from_a_peak(sino_dict, peak):
    sino_allpeaks = sino_dict['sino_allpeaks']
    list_peaks = sino_dict['list_peaks']
    theta = sino_dict['theta']
    idx = list_peaks.index(peak)
    sino = sino_allpeaks[:,:,idx]
    sum_sino = np.sum(sino, 1)   
    return sino, sum_sino, theta

# =============================================================================
# Get the index of the nearest angle in deg
# =============================================================================
def get_idx_angle(theta_array, theta=0):
    theta =theta%360
    x = abs(theta_array-theta).tolist()
    return x.index(min(x))    
    
# =============================================================================
# Get one projection (1d) from the 2D sino, for combinging data
# =============================================================================
def get_proj_from_sino(sino,  idx, width):
    line = np.zeros([1, sino.shape[1]])
    for ii in np.arange(idx-width, idx+width+1):
        line = line + sino[ii,:]
    
    line = line / (width*2+1)    
    ## Normalize
    line = line-np.min(line)
    line = line/np.max(line)
    
    return line

# =============================================================================
# Plot angles on polar coordinate
# =============================================================================
def plot_angles(angles_deg, fignum=100, color='r'):
    angles_deg = np.asarray(angles_deg)
    angles_rad = np.asarray(angles_deg)/180*np.pi
    ones = np.ones(len(angles_deg))
    
    plt.figure(fignum); plt.clf()
    ax = plt.subplot(111, projection='polar')
    ax.bar(angles_rad, ones, width=ones*0.01, color=color, alpha=0.4)
    ax.set_rticks([]) 
    
    for ii, angle in enumerate(angles_rad):
        ax.text(angle, 1, str(angles_deg[ii]), fontweight='bold', color=color)
    
# =============================================================================
#  Find and label peaks   
# =============================================================================
def label_peaks(line_x, line_y):
    peaks, _ = find_peaks(line_y, height=np.mean(line_y)*0.3) #, height=0, width=2, prominence=(0.2, None))
    print(peaks)
    ylim = [np.nanmin(line_y[line_y != -np.inf]), np.nanmax(line_y)]
    yrange = ylim[1]-ylim[0]
    for idx_p, peak in enumerate(peaks):
            plt.plot([line_x[peak], line_x[peak]], ylim, '--', color=rand_color(0.3, 0.9))
            plt.text(line_x[peak], line_y[peak]+(idx_p%10+1)*yrange*0.04, str(np.round(line_x[peak],3)),fontweight='bold')
    
def rand_color(a, b):
    r = b-a
    color = (np.random.random()*r+a, np.random.random()*r+a, np.random.random()*r+a)
    return color
    
    
    
    
    
    
    
    