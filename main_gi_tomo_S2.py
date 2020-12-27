import os, glob, time, sys, tomopy
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from joblib import Parallel, delayed
from scipy import signal

HOME_PATH = '/home/etsai/BNL/Research/GIWAXS_tomo_2019C3/RLi/waxs/'
#HOME_PATH = '/home/etsai/BNL/Research/GIWAXS_tomo_2019C2/RLi2_TOMO_data_201902/RLi2/waxs/'
GI_TOMO_PATH = HOME_PATH+'GI_tomo/'
GI_TOMO_PATH in sys.path or sys.path.append(GI_TOMO_PATH)

import analysis.peaks as peaks
import analysis.tomo as tomo
import analysis.seg as seg
import analysis.util as util
import analysis.io as io

# =============================================================================
# Specify input
# =============================================================================
os.chdir(HOME_PATH)
source_dir = './raw/'
out_dir = './results_tomo/'
infiles = glob.glob(os.path.join(source_dir, '*C8BTBT_0.1Cmin_tomo_*.tiff'))
#infiles = glob.glob(os.path.join(source_dir, '*BTBT_TOMO_test3_*.tiff'))
N_files = len(infiles); print('N_files = {}'.format(N_files))

filename = infiles[0][infiles[0].find('C8BTBT'):infiles[0].find('tomo_')+4]
#filename = infiles[0][infiles[0].find('BTBT'):infiles[0].find('test3_')+6]
print(filename)
if os.path.exists(out_dir) is False: os.mkdir(out_dir)

#### Steps (**only for the first time)
# 1) **Load some 2D data to see peak locations
# 2) **Select/Load roi for peaks 
# 3) **Get peaks from each 2D data. NOTE: slow due to many files
# 4) Generate sinogram for each domain 
# 5) Label peak positons (in deg) for sinos. Use the largest domain (high-intensity peak) to define the set of angles corresponding to one domain.
# 6) Refer to previous step, select domains
# 7) For each domain, generate sinogram and recon
# 8-10) Post-processing/Visualization

run_steps = [10] 
flag_LinearSubBKG = 0
flag_load_peaks = 1 
flag_save_png = 0
flag_save_npy = 0
verbose = 0

## Get ROI for each peak from 2D data (step 2)
filename_peak = './GI_tomo/param/C8BTBT_S2_test_peaks.txt'
#filename_peak = './GI_tomo/param/C8BTBT_S2_peaks.txt'
extra = '_2'
#filename_peak = './GI_tomo/param/BTBT_peaks.txt'


# =============================================================================
# Load all/some data and plot sum
# =============================================================================
if 1 in run_steps:
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
    if flag_save_png:
        fn_out = out_dir+'fig1_'+filename+'_avg.png'
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
        infile_done = util.check_file_exist(infile_done)
        final_img.save(infile_done)     
        
        # Plot qr (after use SciAnalysis on the tiff file)
        if 1:
            fn = './analysis/qr_image/CBTBT_0.1Cmin_tomo_real_data_avg.npz'
            qinfo = np.load(fn)
            qr_image = qinfo['image']
            x_axis = qinfo['x_axis']
            y_axis = qinfo['y_axis']
            extent = (np.nanmin(x_axis), np.nanmax(x_axis), np.nanmin(y_axis), np.nanmax(y_axis))
            plt.figure(11, figsize=[12,8]); plt.clf()
            plt.imshow(np.log10(qr_image), origin='bottom', extent=extent, vmin=0.5, vmax=1.1) 
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
else:
    fn_out = out_dir+'data_avg'
    data_avg = np.load(fn_out+'.npy')

####### Get peak roi from scattering pattern
peak_list = io.read_peak_list(filename_peak)
if 2 in run_steps:     
    #### Plot to define roi
    fig = plt.figure(5, figsize=[12,12]); plt.clf(); 
    plt.title(filename)
    ax = fig.add_subplot(111)
    ax.imshow(np.log10(data_avg), vmin=0.6, vmax=1.1)
    peaks.get_peaks(infiles[0], peak_list, phi_max=360, a=1, verbose=2, FS=17)
    
    ## Save png
    if flag_save_png:
        fn_out = out_dir+'fig5_'+filename+'_peak_roi.png'
        fn_out = util.check_file_exist(fn_out)
        plt.savefig(fn_out, format='png')
    
    ## Save peak_list in npy
    if flag_save_npy:
        fn_out = out_dir+'peak_list'
        fn_out = util.check_file_exist(fn_out)
        np.save(fn_out, peak_list)

        
# =============================================================================
# Get peaks from raw tiff files
# =============================================================================
if 3 in run_steps:     
    t0 = time.time()
    flag_load_parellel = 0  # Sometimes parallel doesn't work..
    if flag_load_parellel:
        with Parallel(n_jobs=3) as parallel:
            results = parallel( delayed(peaks.get_peaks)(infile, peak_list, phi_max=360, verbose=1, flag_LinearSubBKG=flag_LinearSubBKG) for infile in infiles )
    else:
        results = []
        for ii, infile in enumerate(infiles):
            if ii%10==0:
                print('{}'.format(ii))
            temp = peaks.get_peaks(infile, peak_list, phi_max=360, verbose=0, flag_LinearSubBKG=flag_LinearSubBKG)
            results.append(temp)
    print("\nLoad data and define peak roi: {:.0f} s".format(time.time()-t0))
    
    
    # Pass to pd
    df_peaks = pd.DataFrame()
    for ii, df in enumerate(results):
        df_peaks = df_peaks.append(df, ignore_index=True)
    print(df_peaks)
    print(df_peaks.columns)
    
    # Save 
    fn_out = out_dir+'df_peaks_all_subbg{}{}'.format(flag_LinearSubBKG, extra)
    fn_out = util.check_file_exist(fn_out)
    df_peaks.to_csv(fn_out)
 
    # Calculate area
    areas = peaks.calc_area_peakROI(peak_list)

    
# =============================================================================
# Get sino for each peak
#
# Adjust parameters to make sure sinograms look good 
#   1) Check if background/artefacts are more or less removed 
#   2) Check if main features are still there after correction, eg Si has 4 lines
# Adjust parameters to make sure reconstruction looks good
#   1) 00L recon should roughly cover the sample shape
#   2) Choose algo and rotational center
# =============================================================================
if 4 in run_steps: 
    if flag_load_peaks:
        df_peaks = pd.read_csv(out_dir+'df_peaks_all_subbg{}{}'.format(flag_LinearSubBKG,extra))
    
    ## Create sino from pd data    
    list_peaks = [] # Empty if getting all peaks from df_peaks, else specify eg 'sum11L'
    data_sort, sino_dict = tomo.get_sino_from_data(df_peaks, list_peaks=list_peaks, flag_rm_expbg=1, thr=0.2, binary=None) 
    print(sino_dict['list_peaks'])
    sino_sum = tomo.get_sino_sum(sino_dict)
    sino_dict['areas'] = peaks.calc_area_peakROI(peak_list) # Assuming list_peaks are the same as peak_list

    ## Plot sino
    tomo.plot_sino(sino_dict, fignum=10, title_st=filename, vlog10=[0, 5.5])
    if flag_save_png:
        fn_out = out_dir+'fig10_'+filename+'peaks_sino' 
        fn_out = util.check_file_exist(fn_out)
        plt.savefig(fn_out, format='png')
        
    ## Do and plot recon
    recon_all = tomo.get_plot_recon(sino_dict, rot_center=29, algorithms = ['gridrec', 'fbp', 'tv'], fignum=15)
    if flag_save_png:
        fn_out = out_dir+'fig15_'+filename+'peaks_sino_tomo_subbg'+str(flag_LinearSubBKG); 
        fn_out = util.check_file_exist(fn_out)
        plt.savefig(fn_out, format='png')

## Make mask
if 0:  
    plt.figure(16)
    mask = recon_all['sum002_gridrec'][0,:,:].copy()
    mask = (mask-np.min(mask))/np.max(mask)
    plt.imshow(mask)
    im = Image.fromarray(np.uint8((mask*255)))
    im.save('mask_S2_gridrec.png')    
       
    plt.figure(17)
    mask = recon_all['sum002_gridrec'][0,:,:].copy()
    plt.imshow(mask)
   
# =============================================================================
# Label peak positons (in deg) for sinos
#
# Shows the 1D plot (integreated intensity versus degree)
# =============================================================================
if 5 in run_steps: 
    list_peaks = sino_dict['list_peaks']
    flag_log10 = 0 # Use 1 only for plotting
    
    x = {}; jj=0
    N = len(list_peaks[0:-1])
    plt.figure(20, figsize=[15, 8]); plt.clf()
    for ii, peak in enumerate(list_peaks[0:-1]):
    #peak =  'sum20L'
    #if 1:
        sino, sum_sino, theta = tomo.get_sino_from_a_peak(sino_dict, peak) # which peak roi
        if flag_log10: 
            sum_sino = np.log10(sum_sino)    
        plt.subplot(N,1,ii+1)
        plt.plot(theta, sum_sino, color='k');  
        plt.axis('off')     
        if 'b' in peak: color = [0, 0.5, 0] 
        else: color = 'b'
        plt.text(-60, np.max(sum_sino)*0.7, peak, fontsize=10, color=color)
        if ii==0: plt.title(HOME_PATH+', '+filename)
        
        peaks_idx = tomo.label_peaks(theta, sum_sino, onedomain=1)
        
        ## Store peaks and corresponding angles to a df for reconstructing ONE domain
        for angle in theta[peaks_idx]:
            if angle<360: #angle<181: #why
                x[jj] = pd.DataFrame([[angle, peak]], columns=['angle','peak'])
                jj = jj+1
                plt.plot([angle, angle], [0, np.max(sum_sino)*1.1], 'r', linewidth=5, alpha=0.3)
        
    #--- Save to npy
    if flag_save_npy:
        fn_out = out_dir+'angles_onedomain'
        fn_out = util.check_file_exist(fn_out)
        np.save(fn_out, x)
        
    #--- Save to png
    if flag_save_png:
        fn_out = out_dir+'fig20_peak_deg' #+peak
        fn_out = util.check_file_exist(fn_out)
        plt.savefig(fn_out, format='png')

    # =====================================================================
    # Check angles or Load from file    
    # ===================================================================== 
    temp_list = pd.concat(x)
    print(temp_list) #print(list_peaks_angles_orig.sort_values('angle'))
    
    ## Remove peaks not needed for sino
    if 1:
        list_peaks_angles_orig = temp_list[temp_list.peak !='sumSi']
    else:
        temp = np.load('/home/etsai/BNL/Research/GIWAXS_tomo_2019C3/RLi/waxs/results_tomo/list_peaks_angles_orig.npy', allow_pickle=True)
        list_peaks_angles_orig=pd.DataFrame(temp,columns=['angle','peak'])
        
    print('## Compare the list with the figure and drop unwanted peaks.')
    list_peaks_angles_orig = list_peaks_angles_orig[list_peaks_angles_orig.peak !='sumSib']
    #list_peaks_angles_orig = list_peaks_angles_orig.drop([24])   #list_peaks_angles_orig.copy()
    #list_peaks_angles_orig = list_peaks_angles_orig.drop([29]) 
    print(list_peaks_angles_orig)
    tomo.plot_angles(list_peaks_angles_orig['angle'], fignum=21, labels=list_peaks_angles_orig['peak'])    
    #tomo.plot_angles(temp_list['angle'], fignum=23, labels=temp_list['peak'], FS=23)  
    
    if flag_save_png:
        fn_out = out_dir+'fig21_angles' #+peak
        fn_out = util.check_file_exist(fn_out)
        plt.savefig(fn_out, format='png')
    
    np.save(out_dir+'list_peaks_angles_orig.npy', list_peaks_angles_orig)
         
# =============================================================================
# Different domains
# 
# Pick out all (major) domains: 
# Each domain should have eg 11L showing up at certain rotational angles. Assuming the angular sampling is sufficient and that scattering at 11L has sufficient scatering SNR, all domains are captured in the 11L sinogram.
# =============================================================================
if 6 in run_steps: 
    if verbose>1:
        plt.figure(25, figsize=[20, 10]); plt.clf()
        peak_strong = 'sum02L'
        sino, sum_sino, theta = tomo.get_sino_from_a_peak(sino_dict, peak_strong) #choose
        plt.subplot(1,2,1)
        plt.imshow(sino); plt.axis('auto'); plt.ylabel('rotational angle in deg')
        plt.subplot(1,2,2)
        plt.plot(theta, sum_sino);  
        plt.title('sum_sino for {}'.format(peak_strong)); plt.grid()
        peaks_idx = tomo.label_peaks(theta, sum_sino, onedomain=0, fontsize=8)
        print(*theta[peaks_idx], sep=', ')
        
        if flag_save_png:
            fn_out = out_dir+'fig25_angles' #+peak
            fn_out = util.check_file_exist(fn_out)
            plt.savefig(fn_out, format='png')

    ####### Specify domains
    print('## Select the main peaks for reconstruction of different domains. See above for recommendations.')
    #domain_angle_offset = np.asarray([154.0, 157.5, 160.5, 161.5, 165.5, 170.5, 171.5, 188.5, 189.5,]) - 165.5
    #domain_angle_offset = np.append(domain_angle_offset, 12)
    domain_angle_offset = np.arange(85, 138, 0.5)
    #domain_angle_offset= np.append(domain_angle_offset, [1, 179])
    domain_angle_offset = np.sort(domain_angle_offset)
    print('domain_angle_offset = {}'.format(domain_angle_offset))
          

## Do recon for each domain
if 7 in run_steps: 
    recon_all_list = []; sino_all_list = []
    list_peaks_angles = list_peaks_angles_orig.copy()
    
    flag_normal = 1 # 1(normalize max to 1), 2(divided by the ROI area), 3 (binary)
    width = 1
    algo = 'fbp'
    
    if verbose>1: plt.figure(30, figsize=[20, 10]); plt.clf()
    for ii, offset in enumerate(domain_angle_offset):  
        print('offset = {}'.format(offset))
        angles_old = list_peaks_angles_orig['angle'] -112
        angles_new = angles_old + offset
        list_peaks_angles['angle'] = angles_new
    
        ## Get sino
        sino_dm = tomo.get_combined_sino(sino_dict, list_peaks_angles.sort_values('angle'), phi_max=360, width=width, flag_normal=flag_normal, verbose=1)
        ## Plot sino
        if verbose>1: 
            if ii==0:
                title_st = '{}\nflag_normal={}'.format(filename,flag_normal) 
            else:
                title_st = ''
            plt.subplot(2,len(domain_angle_offset),ii+1)
            tomo.plot_sino((sino_dm), theta = sino_dict['theta'], axis_x = sino_dict['axis_x'], title_st=title_st, fignum=-1)
        #plot_angles(list_peaks_angles['angle'], fignum=51)    
        
        ## Tomo recon
            plt.subplot(2,len(domain_angle_offset),len(domain_angle_offset)+ii+1)
            title_st = '[{}] ori={}$^\circ$'.format(ii,offset)
        else: 
            title_st = ''
        temp = tomo.get_plot_recon(sino_dm, theta = sino_dict['theta'], rot_center=29, algorithms = [algo], title_st=title_st, fignum=None, colorbar=True)
        sino_all_list.append(sino_dm)
        recon_all_list.append(np.squeeze(temp['_{}'.format(algo)]))
       
    if verbose>1 and flag_save_png:     
        fn_out = out_dir+'fig30_recon'
        fn_out = util.check_file_exist(fn_out)
        plt.savefig(fn_out, format='png')
   
# =============================================================================
# Overlap three domains spatially
# =============================================================================
if 8 in run_steps:     
    domains_use = [0, 1, 2]   #overlay these domains
    rgb = 'RGBWCMY'
    channel=0; overlay = []
    
    plt.figure(35, figsize=[20,10]); plt.clf()
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
        fn_out = out_dir+'fig35_recon_overlay{}{}{}'.format(domains_use[0], domains_use[1], domains_use[2])
        fn_out = check_file_exist(fn_out)
        plt.savefig(fn_out, format='png')


# =============================================================================
# Plot all recons after threshold
# =============================================================================
if 9 in run_steps:
    recon_merged = np.zeros([recon_all_list[0].shape[0], recon_all_list[0].shape[1]])
    Ndomain = len(domain_angle_offset)
    
    plt.figure(40, figsize=[20,10]); plt.clf()
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
if 10 in run_steps:
    domains_use = np.arange(0, len(recon_all_list))
    recon_all_list_normal = []
    
    for ii in domains_use:      
        recon = recon_all_list[ii]
        recon_thr = recon.copy()
        recon_thr[recon<np.max(recon)*0.1]=0
        recon_all_list_normal.append(recon_thr/np.max(1))
        
    mask = (recon!=0).astype(float)
    mask_nan = mask.copy()
    mask_nan[mask==0] = np.nan
    
    temp_angle = domain_angle_offset[domains_use]
    domains_recon = mask_nan*temp_angle[np.argmax(recon_all_list_normal,0)]
    
    ##------ Load mask
    plt.figure(45); plt.clf()
    if 1:
        x = np.asarray(Image.open("./mask_S2b.png").convert("L").resize((50,50)))
        plt.imshow(x, alpha = 1, cmap='binary')
        x = x.astype('float')
        x[x<3] = 0
        x[x>0] = 1.0
        x[x==0] = np.nan
    else:
        x = 1
    
    plt.imshow(domains_recon*x, cmap='twilight', alpha = 0.95, vmin=0, vmax=180)
    cbar = plt.colorbar(fraction=0.03, pad=0.0, aspect=20) 
    util.plot_quiver(domains_recon*x)
    plt.axis('off')
    plt.plot([5, 10], [45, 45], linewidth=4, color='k')
    plt.text(4.8, 43.7, '1mm', color='k', fontweight='bold', fontsize=10)
    
    
    ## Save 
    if flag_save_png:    
        fn_out = out_dir+'fig45_domains_recon'
        fn_out = util.check_file_exist(fn_out)
        plt.savefig(fn_out, format='png')

    if flag_save_npy:         
        fn_out = out_dir+'domains_recon.npy'
        np.save(fn_out, domains_recon)
        
        fn_out = out_dir+'sino_all_list.npy'
        np.save(fn_out, sino_all_list)
        
        rot_angles = np.asarray(list_peaks_angles_orig.sort_values('angle')['angle'])
        fn_out = out_dir+'rot_angles.npy'
        np.save(fn_out, rot_angles)







