#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python implementation of Consensus Equilibrium (CE) for combining multiple 
SNR-mismatched U-net denoisers for SNR matched denoising  

Haonan Lin
June 2024

"""

from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
import matplotlib.pyplot as plt


from tifffile import imread
from csbdeep.utils import  plot_some
from csbdeep.io import save_tiff_imagej_compatible
from csbdeep.models import CARE


# Load target image 

datapath = './data/NACE_20230610/Fig1_data/5avg_BSA/'
filename = 'BSA_17.tif'
y = imread(datapath + filename)

# Set noise level for ns and no
noise_level = 473

# ns noise levels for each averaging pair

ns = np.array([862, 512, 362, 242])

# Initialization
max_itr         = 10
num_denoiser    = 4
[rows, cols]    = np.shape(y)
zhat_init       = np.zeros([rows, cols,num_denoiser+1])
zhat_init[:,:,0]= y
#mse_est         = np.zeros([1,num_denoiser])
#psnrFv          = np.zeros([max_itr+1,num_denoiser+1])    # individual denoiser
#psnrzhat        = np.zeros([max_itr+1,1])                 # CE solution

# Denoisers

model = [CARE(config=None, name='CARE_NAC_5_1_pair_BSA', basedir='models'), 
         CARE(config=None, name='CARE_NAC_5_2_pair_BSA', basedir='models'),
         CARE(config=None, name='CARE_NAC_5_3_pair_BSA', basedir='models'),
         CARE(config=None, name='CARE_NAC_5_4_pair_BSA', basedir='models')]
axes  = 'YX'

# Compute weights
p1 = np.exp(- (ns-noise_level)**2 / (2*(250)**2)  )
p1 = p1/np.sum(p1)
p  = np.append(np.sum(p1),p1)
p  = p/np.sum(p)

# Initialize estimates & Calculate baseline estimate
zhat_baseline = 0
for i in range(num_denoiser):
    xhat                = model[i].predict(y, axes, n_tiles=16)
    zhat_init[:,:,i+1]  = xhat
    zhat_baseline       = zhat_baseline + p1[i]*zhat_init[:,:,i+1]

# CE main routine
print ('========================================== \n')
print ('Running Consensus Equilibrium: ', filename, '\n')

gamma   = 0.5
Fv      = np.zeros([rows, cols,num_denoiser+1])
rho     = 1
w       = zhat_init
zhat    = np.sum(np.multiply(np.tile(np.reshape(p,(1,1,num_denoiser+1)),[rows,cols,1]),w),axis=2)

for itr in range(max_itr):
    # ====== Update v ======#
    Gw  = zhat
    Gw  = np.repeat(Gw[...,None],num_denoiser+1,axis=2)
    v   = 2*Gw - w
    
    # ====== Update w ======#
    Fv[:,:,0] = (y+rho*v[:,:,0]) / (1+rho)
    for i in range(num_denoiser):
        Fv[:,:,i+1] = model[i].predict(v[:,:,i+1], axes, n_tiles=16)
    w   = (1-gamma)*w + gamma*(2*Fv-v)
    
    # ====== Compute zhat ======#
    zhat = np.sum(np.multiply(np.tile(np.reshape(p,(1,1,num_denoiser+1)),[rows,cols,1]),w),axis=2)
    
    print ('iter ', i+1, '\n')

# ====== Output CE results ====== #
save_tiff_imagej_compatible('results/%s_%s' % ('CE',filename), zhat, axes )

plt.figure(figsize=(16,10))
plot_some(np.stack([y,zhat]),
          title_list=[['low SNR (maximum projection)','Denoising (maximum projection)']], 
          pmin=2,pmax=99.8);








