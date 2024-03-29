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
def get_sino_from_data(data, list_peaks=[], flag_rm_expbg=1, thr=None, binary=None, flag_align=0):
    
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
            print('NOTE: flag_rm_expbg=1, this substracts the normalized sumBKG0 (normalized to the projection value at which the sumBKG0 is max) from the projection')
            proj = proj - proj_bkg*proj[bkg_max_idx]         


        print('# Checking sino & Fill empty data frame')
        proj[proj<5] = 0
        for xx in np.arange(1, proj.shape[1]-1):
            #print('{}'.format(np.mean(proj)*0.05))
            if np.mean(proj[:,xx])<np.mean(proj)*0.05:
                proj[:,xx] = (proj[:,xx-1] + proj[:, xx+1])/2
   
        
        if thr is not None:
            print('NOTE: proj[proj<np.median(proj)*thr] = 1')
            proj[proj<np.median(proj)*thr] = 1
            
        if thr is not None and binary is not None: 
                print('NOTE: proj[proj>=thr] = binary')
                proj[proj>=thr] = binary
            
        if flag_align:                
            print('NOTE: Manually align sino')
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
    
    if fignum>0: 
        plt.figure(fignum, figsize=[12,12]); plt.clf()    
    Npeaks =  sino_allpeaks.shape[2]
    
    for ii in np.arange(0, sino_allpeaks.shape[2]):
        sino = sino_allpeaks[:,:,ii]
        sum_sino = np.sum(sino, 1)   
        peak = list_peaks[ii] if list_peaks!=[] else ''
        
        if fignum>0: 
            plt.subplot(3,Npeaks,ii+1)

        plt.imshow(sino, cmap='gray', aspect='auto', vmin=0, vmax=1) #, extent = [axis_x[0], axis_x[-1], theta[-1], theta[0]])
        plt.axis('off')
        
        if fignum>0:
            if ii==0: 
                plt.title('{}\n{}'.format(title_st, peak), fontweight='bold')
                plt.axis('on');
            elif ii%2: plt.title('{}'.format(peak), fontweight='bold')
            else: plt.title('{}'.format(peak))
            v1 = np.linspace(sino.min(), sino.max(), 2, endpoint=True)
            cb = plt.colorbar(orientation='horizontal', pad=0.05, ticks=v1)
            cb.ax.set_xticklabels(["{:.1e}".format(v) if v>1 else "" for v in v1])
        else:
            plt.title(title_st, fontweight='bold')
        
        if fignum>0: 
            ax1 = plt.subplot(3,Npeaks,Npeaks+ii+1)
            plt.plot(sum_sino, theta)
            peaks_idx = label_peaks(theta, sum_sino, onedomain=1, axis_flip=1)
            ## Overlay with log10 
            ax2 =ax1.twiny()
            ax2.plot(np.log10(sum_sino), theta, 'r', alpha=0.3);
            ax2.axis('off'); ax1.axis('off')
            
            plt.subplot(3,Npeaks,Npeaks*2+ii+1)            
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
def get_plot_recon(sino_data, theta = [], rot_center=0, algorithms = ['art', 'gridrec', 'fbp'], title_st=[], fignum=40,  cmap='jet', colorbar=False):
    if type(sino_data)==dict:
        sino_allpeaks = sino_data['sino_allpeaks']
        theta = sino_data['theta']
        axis_x = sino_data['axis_x']
        list_peaks = sino_data['list_peaks']
    else:
        sino_data = np.asarray(sino_data)
        sino_allpeaks = np.reshape(sino_data, (sino_data.shape[0], sino_data.shape[1], 1))
        list_peaks = []
    theta_rad = theta/180*np.pi 
        
    if fignum is not None:
        if fignum>0: 
            plt.figure(fignum, figsize=[12,12]); plt.clf()    
    Npeaks =  sino_allpeaks.shape[2]  
    recon_all = {}
    for ii in np.arange(0, sino_allpeaks.shape[2]):
        sino = sino_allpeaks[:,:,ii]
        sino = np.reshape(sino, (sino.shape[0], 1, sino.shape[1]))
        peak = list_peaks[ii] if list_peaks!=[] else ''
        
        #proj = tomopy.minus_log(proj)
        #flat = np.ones((1,1,sino.shape[2]))
        #dark = np.zeros((1,1,sino.shape[2]))
        #sino = tomopy.normalize(sino, flat, dark) # (sino-dark)/(flat-dark)
        
        ## Get rotational center if not specified
        cen_init = sino.shape[2]/2
        if rot_center<=0:
            rot_center = tomopy.find_center(sino, theta_rad, init=cen_init, ind=0, tol=0.1)
            print('Rotational center: {}'.format(rot_center))
            if (rot_center>cen_init+10) or (rot_center<cen_init-10):
                rot_center = sino.shape[2]/2

        ## Tomo reconstruction 
        # Keyword "algorithm" must be one of ['art', 'bart', 'fbp', 'gridrec', 'mlem', 'osem', 'ospml_hybrid', 
        # 'ospml_quad', 'pml_hybrid', 'pml_quad', 'sirt', 'tv', 'grad'], or a Python method.
        #recon = tomopy.recon(proj, theta, center=rot_center, algorithm=tomopy.lprec, lpmethod='tv', ncore=1, num_iter=512, reg_par=5e-4)
        
        #rot_center = cen_init
        #plt.figure(50)
        print('rot_center = {}'.format(rot_center))
        for jj, algo in enumerate(algorithms):
            recon = tomopy.recon(sino, theta_rad, center=rot_center, algorithm=algo)
            recon = tomopy.circ_mask(recon, axis=0, ratio=0.95)
            
            recon_all[peak+'_'+algo] = recon
            
            #recon[recon==0] = np.nan
            if fignum is not None:
                if fignum>0: 
                    plt.subplot(len(algorithms), Npeaks, (jj)*Npeaks+ii+1)
                plt.imshow(recon[0, :,:], cmap=cmap) #, vmin=0, vmax=1.5e7)
                if title_st==[]:
                    if ii%2: plt.title('{}\n{}, cen{:.1f}'.format(peak, algo, float(rot_center)), fontweight='bold')
                    else: plt.title('{}\n{}, cen{:.1f}'.format(peak, algo, float(rot_center)))
                else:
                    plt.title(title_st)
    
                if colorbar:
                    v1 = np.linspace(recon.min(), recon.max(), 2, endpoint=True)
                    cb = plt.colorbar(orientation='horizontal', pad=0.05, ticks=v1)
                    cb.ax.set_xticklabels(["{:.0f}".format(v) if v>1 else "" for v in v1])
                plt.axis('off')
            
    return recon_all

# =============================================================================
# Combine data from different peaks into one sino for a domain
# =============================================================================
def get_combined_sino(sino_dict, list_peaks_angles, phi_max=360, width=0, flag_normal=1, verbose=0):
    sino_allpeaks = sino_dict['sino_allpeaks']
    theta = sino_dict['theta']
    areas = sino_dict['areas']
    
    peaks = np.asarray(list_peaks_angles['peak'])
    angles = np.asarray(list_peaks_angles['angle'])
    
    if verbose>0: print('------')
    sino_dm = np.zeros([sino_allpeaks.shape[0], sino_allpeaks.shape[1]])
    count_dm = np.zeros([sino_allpeaks.shape[0], sino_allpeaks.shape[1]])
    
    for ii in np.arange(0,len(list_peaks_angles)):
        ## Get sino for a peak (eg 110)
        sino, _, _ = get_sino_from_a_peak(sino_dict, peaks[ii])  
        idx = get_index_for_peak(sino_dict, peaks[ii])
        ## Get the projection from the sino at certain angle
        angle = angles[ii]
        if verbose>0: print('angle = {}, peak = {}, area {}'.format(np.mod(angle, phi_max), peaks[ii], areas[idx]))        
        angle_idx = get_idx_angle(theta, theta=np.mod(angle, phi_max))
        temp = get_proj_from_sino(sino,  angle_idx, width, flag_normal=flag_normal)  # get the projection at the angle
        if phi_max==180:
            if angle>phi_max or angle<0:
                temp = np.flip(temp)
        
        ## Normalize projection
        if flag_normal==2:
            temp = temp/areas[idx]*100  
        elif flag_normal==3:
            thr = np.max(temp.copy())*0.5
            temp[temp<thr] = 0
            temp[temp>thr] = 1
           
        ## Populate domain sino with projections
        sino_dm[angle_idx-width:angle_idx+width+1, :] = sino_dm[angle_idx-width:angle_idx+width+1, :]+temp
        count_dm[angle_idx-width:angle_idx+width+1, :] = count_dm[angle_idx-width:angle_idx+width+1, :]+1

    count_dm[count_dm==0] = 1
    sino_dm = sino_dm/count_dm
    sino_dict['sino_dm'] = sino_dm
    if verbose>0: print('------')
    
    return sino_dm                  

# =============================================================================
# Get index for a peak (eg 'sum11L')
# =============================================================================
def get_index_for_peak(sino_dict, peak):
    list_peaks = sino_dict['list_peaks']
    idx = list_peaks.index(peak)
    return idx    

# =============================================================================
# Get the sino for a certain peak (eg 'sum11L')
# =============================================================================
def get_sino_from_a_peak(sino_dict, peak):
    sino_allpeaks = sino_dict['sino_allpeaks']
    idx = get_index_for_peak(sino_dict, peak)
    sino = sino_allpeaks[:,:,idx]
    sum_sino = np.sum(sino, 1)   
    theta = sino_dict['theta']
    return sino, sum_sino, theta
  
                         
# =============================================================================
# Get the index of the nearest angle in deg
# =============================================================================
def get_idx_angle(theta_array, theta=0):
    theta = theta%360
    x = abs(theta_array-theta).tolist()
    return x.index(min(x))    
    
# =============================================================================
# Get one projection (1d) from the 2D sino, for combinging data
# =============================================================================
def get_proj_from_sino(sino,  idx, width, flag_normal=1):
    line = np.zeros([1, sino.shape[1]])
    
    w = 0
    for ii in np.arange(idx-width, idx+width+1):
        if ii > 0 and ii < sino.shape[0]:
            line = line + sino[ii,:]
            w = w+1
    line = line / w   
    
    ## Normalize
    if flag_normal>=1:
        line = line-np.min(line)
        if np.max(line)>0:
            line = line/np.max(line)
    
    return line

# =============================================================================
# Plot angles on polar coordinate
# =============================================================================

def plot_angles(angles_deg, fignum=100, color='r', labels=[], FS=15, theory=0):
    angles_deg = np.asarray(angles_deg)
    angles_rad = np.asarray(angles_deg)/180*np.pi
    
    angles_deg = [round(xx,1) for xx in angles_deg] # Round up
    ones = np.ones(len(angles_deg))
    
    plt.figure(fignum); plt.clf()
    ax = plt.subplot(111, projection='polar')
    ax.bar(angles_rad, ones*0.8, width=ones*0.01, color=color, alpha=0.8)
    ax.set_rticks([]) 
    ax.set_xticklabels([])
    
    green = [0, 0.6, 0] # [0, 0.6, 0]; 
    if theory==1:
        green = [0, 0, 0.9]

    FW1='normal'; FW='bold'
    if type(labels) is not list:
        labels = labels.values.tolist()
        
    if 'sum' in labels[0]: 
        s = 3
    else: s = 0
    
    for ii, angle in enumerate(angles_rad):
        label = labels[ii]
               
        ## Label angles
        if 'Si' in label:
            ax.bar(angle, ones, width=ones*0.01, color='k', alpha=0.6)
            tt = ax.text(angle, 1.24, str(angles_deg[ii]), color='k', fontsize=FS-4, fontweight=FW1, ha='center', va='center')
        elif '0' in label: # and label[-1]=='0':
                ax.bar(angle, ones*0.8, width=ones*0.01, color=green, alpha=0.8)
                tt = ax.text(angle, 0.9, str(angles_deg[ii]), color=green, fontsize=FS-4,fontweight=FW1, ha='center', va='center')
        else:
            tt = ax.text(angle, 0.86, str(angles_deg[ii]), color=color, fontsize=FS-4, fontweight=FW1, ha='center',va='center')
            
        if np.cos(angle)<=0.01: rotate=angle+np.pi
        else: rotate = angle
        tt.set_rotation(rotate/np.pi*180)
        
        ## Label peak
        if len(labels)>0:
            if 'Si' in label:
                tt2 = ax.text(angle, 1.34, label[s:], color='k', fontsize=FS-5, fontweight=FW, ha='center',va='center')
            elif '0' in label: #and label[-1]=='0':
                tt2 = ax.text(angle, 1.05-theory*0.05, label[s:], color=green, fontsize=FS, fontweight=FW, ha='center',va='center')
            else:
                tt2 = ax.text(angle, 1.0, label[s:], color=color,fontsize=FS, fontweight=FW, ha='center', va='center')

        #tt2.set_rotation(rotate/np.pi*180)
            
    plt.show()

# =============================================================================
#  Find and label peaks   
#  onedomain = 1 to (attempt to) find peaks corresponding to the same domain
# =============================================================================
def label_peaks(line_x, line_y, onedomain=0, axis_flip=0, fontsize=9, color=[0.2, 0.2, 0.2]):
    
    if onedomain:
        peaks, _ = find_peaks(line_y, height=np.mean(line_y)*1.7, distance=38/(line_x[1]-line_x[0])) #prominence=(0.2, None)) #width=2,
    else:
        peaks, _ = find_peaks(line_y, height=np.mean(line_y)*0.5)

    ylim = [np.nanmin(line_y[line_y != -np.inf]), np.nanmax(line_y)]
    yrange = ylim[1]-ylim[0]
    
    for idx_p, peak in enumerate(peaks):
        if axis_flip==0:
            #plt.plot([line_x[peak], line_x[peak]], ylim, '--', color=rand_color(0.4, 0.5)) #rand_color(0.3, 0.9)
            plt.text(line_x[peak], line_y[peak]*0.6+(idx_p%5+1)*yrange*0.05, str(np.round(line_x[peak],3)),fontweight='bold', fontsize=fontsize, color=color)
        else:
            #plt.plot(ylim, [line_x[peak], line_x[peak]], '--', color=rand_color(0.4, 0.5))
            plt.text(line_y[peak]*0.9, line_x[peak]*1.1, str(np.round(line_x[peak],3)),fontweight='bold', fontsize=fontsize, color=color)
    
    if axis_flip: 
        plt.gca().invert_yaxis()
        
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    plt.show()
    
    return peaks

   
# =============================================================================
# Fill (peak) sino
# =============================================================================
def fill_sino(sino, thr = 2, verbose=0):
    if verbose>0:
        print('# Fill empty projection')
    sino_filled = sino.copy()
    for ii in np.arange(1, sino.shape[0]-1):
        if np.sum(sino[ii,:]) < thr:
            row1 = 0; row2 = 0
            x1 = 0; x2 = 0
            while np.sum(row1) < thr and ii-x1>1:
                x1 = x1+1
                row1 = sino[ii-x1]
            while np.sum(row2) < thr and ii+x2<sino.shape[0]-1:
                x2 = x2+1
                row2 = sino[ii+x2]
            
            sino_filled[ii,:] = (row1*x2 + row2*x1)/(x1+x2)
    return sino_filled                
 

    
    
    
    
    
    
    
    
    
    
    
    
    
    
               
    
