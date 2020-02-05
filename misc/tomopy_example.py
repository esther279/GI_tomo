#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tomo example from tomopy
"""

import tomopy
import dxchange
import matplotlib.pyplot as plt

#fname = '../../tomopy/data/tooth.h5'
fname = '/home/etsai/Globus/tomobank/phantom_00007/phantom_00007.h5'
#fname = '/home/etsai/Globus/tomobank/tomo_00001_to_00006/tomo_00002/tomo_00002.h5'

start = 0
end = 2
proj, flat, dark, theta = dxchange.read_aps_32id(fname, sino=(start, end))

plt.figure(1); plt.clf()
plt.imshow(proj[:, 0, :], cmap='Greys_r',aspect='auto')
plt.show()

# modify sino
print(proj.shape)
#proj[0:200, 0, :] = 0
#plt.imshow(proj[:, 0, :], cmap='Greys_r',aspect='auto')
#plt.show()




### Get center
theta = tomopy.angles(proj.shape[0])
proj = tomopy.normalize(proj, flat, dark) # (proj-dark)/(flat-dark)
#rot_center = tomopy.find_center(proj, theta, init=290, ind=0, tol=0.5)
rot_center = tomopy.find_center(proj, theta, init=None, ind=0, tol=0.1)
print(rot_center)


#proj = tomopy.minus_log(proj)





### Tomo reconstruction
# Keyword "algorithm" must be one of ['art', 'bart', 'fbp', 'gridrec', 'mlem', 'osem', 'ospml_hybrid', 
# 'ospml_quad', 'pml_hybrid', 'pml_quad', 'sirt', 'tv', 'grad'], or a Python method.

#recon = tomopy.recon(proj, theta, center=rot_center, algorithm='gridrec')
#recon = tomopy.recon(proj, theta, center=rot_center, algorithm=tomopy.lprec, lpmethod='tv', ncore=1, num_iter=512, reg_par=5e-4)
recon = tomopy.recon(proj, theta, center=rot_center, algorithm='art')


recon = tomopy.circ_mask(recon, axis=0, ratio=0.95)

plt.figure(2); plt.clf()
plt.imshow(recon[0, :,:], cmap='Greys_r')
plt.colorbar()
plt.show()



