#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, glob, time, sys
import numpy as np
import matplotlib.pyplot as plt
import copy
import tomopy

def get_sino_from_data(data, list_peaks=[], flag_rm_expbg=1, flag_thr=0):
    
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
        
        if flag_thr:
            thr = np.median(proj)*7
            print('thr = {}'.format(thr))
            proj[proj<thr] = 0
            #proj[proj>=thr] = 1
        
        sino_allpeaks[:,:,ii] = proj
        
    sino_dict = {}
    sino_dict['sino_allpeaks'] = sino_allpeaks
    sino_dict['theta'] = theta
    sino_dict['axis_x'] = axis_x
    sino_dict['list_peaks'] = list_peaks
    
    return data_sort, sino_dict

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
    
    plt.figure(fignum, figsize=[12,12]); plt.clf()    
    Npeaks =  sino_allpeaks.shape[2]
    for ii in np.arange(0, sino_allpeaks.shape[2]):
        sino = sino_allpeaks[:,:,ii]
        peak = list_peaks[ii] if list_peaks!=[] else ''
        
        plt.subplot(2,Npeaks,ii+1)
        plt.imshow(sino, cmap='jet', aspect='auto') #, extent = [axis_x[0], axis_x[-1], theta[-1], theta[0]])
        plt.axis('off')
        if ii==0: 
            plt.title('{}\n{}'.format(title_st, peak), fontweight='bold')
            plt.axis('on');
        elif ii%2: plt.title('{}'.format(peak), fontweight='bold')
        else: plt.title('{}'.format(peak))
        v1 = np.linspace(sino.min(), sino.max(), 2, endpoint=True)
        cb = plt.colorbar(orientation='horizontal', pad=0.05, ticks=v1)
        cb.ax.set_xticklabels(["{:.1e}".format(v) if v>0 else "" for v in v1])
        
        plt.subplot(2,Npeaks,Npeaks+ii+1)
        plt.imshow(np.log10(sino), cmap='jet', aspect='auto', extent = [axis_x[0], axis_x[-1], theta[-1], theta[0]], vmin=vlog10[0], vmax=vlog10[1])
        if ii==0: 
            plt.axis('on'); plt.title('log10')
            plt.xlabel('pos_x (mm)')
            plt.ylabel('pos_phi (deg)')    
        else: plt.axis('off')
        if ii==sino_allpeaks.shape[2]-1:
            plt.colorbar(orientation='horizontal', pad=0.01)    

def get_recon(sino_data, theta = [], rot_center=10, algorithms = ['art', 'gridrec', 'fbp'], fignum=40):
    if type(sino_data)==dict:
        sino_allpeaks = sino_data['sino_allpeaks']
        theta = sino_data['theta']
        axis_x = sino_data['axis_x']
        list_peaks = sino_data['list_peaks']
    else:
        sino_data = np.asarray(sino_data)
        sino_allpeaks = np.reshape(sino_data, (sino_data.shape[0], sino_data.shape[1], 1))
        list_peaks = []
    
    plt.figure(fignum, figsize=[12,12]); plt.clf()    
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
        
        cen_init = sino.shape[2]/2
        if rot_center<=0:
            rot_center = tomopy.find_center(sino, theta, init=cen_init, ind=0, tol=0.1)
            print('Rotational center: {}'.format(rot_center))
            if (rot_center>cen_init+5) or (rot_center<cen_init-5):
                rot_center = sino.shape[2]/2
                
        
        ##########################################
        # Tomo reconstruction
        ##########################################      
        # Keyword "algorithm" must be one of ['art', 'bart', 'fbp', 'gridrec', 'mlem', 'osem', 'ospml_hybrid', 
        # 'ospml_quad', 'pml_hybrid', 'pml_quad', 'sirt', 'tv', 'grad'], or a Python method.
        
        #recon = tomopy.recon(proj, theta, center=rot_center, algorithm=tomopy.lprec, lpmethod='tv', ncore=1, num_iter=512, reg_par=5e-4)
        
        #rot_center = cen_init
        #plt.figure(50)
        for jj, algo in enumerate(algorithms):
            recon = tomopy.recon(sino, theta, center=rot_center, algorithm=algo)
            recon = tomopy.circ_mask(recon, axis=0, ratio=0.95)
            plt.subplot(len(algorithms), Npeaks, (jj)*Npeaks+ii+1)
            plt.imshow(recon[0, :,:], cmap='jet') #, vmin=0, vmax=1.5e7)
            if ii%2: plt.title('{}\n{}, cen{:.1f}'.format(peak, algo, float(rot_center)), fontweight='bold')
            else: plt.title('{}\n{}, cen{:.1f}'.format(peak, algo, float(rot_center)))
            #plt.colorbar(); plt.show();
            recon_all[peak+'_'+algo] = recon
            
    return recon_all
    

def get_combined_sino(sino_dict_dm, list_peaks=[], angle0=0, peak_angles_offset=[0], width=0):
    sino_allpeaks = sino_dict_dm['sino_allpeaks']
    theta = sino_dict_dm['theta']
    
    sino_dm = np.zeros([sino_allpeaks.shape[0], sino_allpeaks.shape[1]])
    for ii in np.arange(0, sino_allpeaks.shape[2]):
        sino = sino_allpeaks[:,:,ii]
        peak = list_peaks[ii] if list_peaks!=[] else ''
        
        angle =  angle0 + peak_angles_offset[ii]
        print('angle = {}'.format(angle))
        angle_idx = get_idx_angle(theta, theta=angle)
        sino_dm[angle_idx,:] = get_proj_from_sino(sino,  angle_idx, width) 

    sino_dict_dm['sino_dm'] = sino_dm
    return sino_dm
 
def get_idx_angle(theta_array, theta=0):
    x = (theta_array==theta).tolist()
    return x.index(max(x))    
    

def get_proj_from_sino(sino,  idx, width):
    line = np.zeros([1, sino.shape[1]])
    for ii in np.arange(idx-width, idx+width+1):
        line = line + sino[ii,:]
    
    line = line / (width*2+1)
    
    return line

