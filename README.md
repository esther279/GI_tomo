# GI_tomo
https://github.com/esther279/GI_tomo.git

Ref: Tsai, Esther HR, Yu Xia, Masafumi Fukuto, Y-L. Loo, and Ruipeng Li. "Grazing-incidence X-ray diffraction tomography for characterizing organic thin films." Journal of Applied Crystallography 54, no. 5 (2021): 1327-1339. https://doi.org/10.1107/S1600576721007184 

Scripts:
1. Typical tomographic reconstruction example: **tomopy_example.py**
2. Simulation: **main_sim.py** (for limited angles), **main_sim_peak.py** (for sample with different orientations and sino for a peak)
3. Reconstruction for GI data: **main_gi_tomo.py**
    1) Load all or some 2D data (TIFF), plot the average 'data_avg' to see where the peak locations are
    2) **Select roi** for peaks (e.g. [[575, 252], [60, 10], 'sum002'] for location, roi size, name), store in a TXT file, and load with 'peak_list = io.read_peak_list(filename_peak)'
    3) Get peaks from all each 2D data (TIFF) with 'peaks.get_peaks(infile, peak_list, ..)', store output in pandas data frame 'df_peaks'. NOTE this step is bit slow
    4) Generate sinogram for each domain with 'sino_dict = tomo.get_sino_from_data(df_peaks, ..)'
    5) Label peak positons (in deg) for sinos with 'sino, sum_sino, .. = tomo.get_sino_from_a_peak(sino_dict, peak)'. Use the largest domain (high-intensity peak) to **define** the set of angles corresponding to one domain. 
    6) Refer to previous step, manually select orientations for domains with 'domain_angle_offset = np.asarray([21, 51, 65, 172])'
    7) For each angle (ie domain), generate sinogram with 'sino_dm = tomo.get_combined_sino(sino_dict, ..)', results stored in 'sino_all_list and 'recon_all_list'
    
