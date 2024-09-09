# NACE
Noisy-As-Clean with consensus equilibrium (NACE) is a self-supervised, non-independent denoising algorithm for 2D microscopic images. The raw data needs to be acquired as a sequence of repeated measurements. The target noisy image is generated by averaging the entire sequence.

# Prerequesite
The code relies on the CSBDeep Python package (https://github.com/CSBDeep/CSBDeep) for U-net denoiser implementation. A copy of the csbdeep package is included in the folder.

Software dependencies and tested versions: Python 3 (3.9.12), Tensorflow 2 (2.11.0) with GPU support (CUDA (9.1) and cuDNN (8.8.0))

CSBDeep copyright:
BSD 3-Clause License
Copyright (c) 2018, Uwe Schmidt, Martin Weigert
All rights reserved.

# User Guide
The NACE algorithm aims to perform self-supervised, learning-based denoising of 2D microscopic images without noise statistics assumption and is robust against non-independent noise.

## Step 1: Generating noisier-noisy images pairs
NACE requires the raw data to be a sequence of N repeated measurements (t1, t2, ..., tN) and the target noisy image is the average of the entire sequence, denoted as **y** = mean (t1, t2, ..., tN). 

By reducing the number of averaged frames to k (k<N), we can generate a series of noisier images, denoted as **zk** = mean (t1, t2, ..., tk). 

In the provided example data, we have 5avg as the noisy image **y**, and 1avg, 2avg, 3avg, 4avg as noisier images with different levels of added noise (**z1, z2, z3, z4**).

In the first script, **1_prepare_training_data_NACE.py**, we prepare a series of training data with **zk-y** image pairs. Here, we show one example of **z1-y**. Line 28 can be modified to generate the remaining training pairs.

## Step 2: Training individual Noisy As Clean (NAC) denoisers using different noisier-noisy image pairs

In the second script, **2_training_NACE.py**, we can train a series of NAC denoisers by taking **zk** and **y** as inputs and outputs, respectively.

Again, the script shows the training of **z1-y** NAC denoiser. Lines 24 and 56 need to be modified to change the input data and model name for other NACs.

## Step 3: Using consensus equilibrium (CE) to perform SNR-matched denoising by combining several NAC denoisers

In the third script, **3_prediction_NACE.py**, we use the consensus equilibrium framework to combine a series of SNR-mismatched NAC denoisers and generate SNR-matched denoising performance.

Line 30 is the measured noise level for y. Added noise levels are shown in Line 34. These parameters need to be calculated for different datasets.

The script will generate an optimized output iteratively. The number of iterations can be adjusted in line 37.
