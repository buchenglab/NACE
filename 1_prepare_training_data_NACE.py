
"""
Generate noisier-noisy image pairs for CARE-NAC denoising
Image pairs are generated directly from consecutive acquired raw SRS images
Haonan Lin 
June 2024

"""

from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np

import matplotlib.pyplot as plt

from tifffile import imread

from csbdeep.utils import download_and_extract_zip_file, plot_some,axes_dict
from csbdeep.data import RawData, create_patches, no_background_patches


# Create a RawData object, which defines how to get the pairs of low/high SNR stacks 
# and the semantics of each axis (e.g. which one is considered a color channel, etc.).

#raw_data = RawData.from_arrays(x, y, axes = "ZYX")

raw_data = RawData.from_folder (
    basepath    = './data/SKOV3',
    source_dirs = ['1avg_BSA'],
    target_dir  = '5avg_BSA',
    axes        = 'YX',
)

# Now generate some 2D patches
X, Y, XY_axes = create_patches (
    raw_data            = raw_data,
    patch_size          = (128,128),
    n_patches_per_image = 64,
    patch_filter  = no_background_patches(threshold=0.8, percentile=97),
    save_file           = './data/SKOV3/NAC_5-1_avg_pairs_BSA.npz',
)

assert X.shape == Y.shape
print("shape of X,Y =", X.shape)
print("axes  of X,Y =", XY_axes)


# shows the maximum projection of some of the generated patch pairs 
for i in range(2):
    plt.figure(figsize=(16,4))
    sl = slice(8*i, 8*(i+1)), 0
    plot_some(X[sl],Y[sl],title_list=[np.arange(sl[0].start,sl[0].stop)])
    plt.show()
None;
