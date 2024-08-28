#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Training a CARE network for NAC denoising of SRS images
Haonan Lin
June 2024

"""

from __future__ import print_function, unicode_literals, absolute_import, division
import matplotlib.pyplot as plt
#from tifffile import imread

from csbdeep.utils import axes_dict, plot_some, plot_history
from csbdeep.utils.tf import limit_gpu_memory
from csbdeep.io import load_training_data
from csbdeep.models import Config, CARE

#limit_gpu_memory(fraction=1/2)

# Load training data and validation data, use 10% validation data
(X,Y), (X_val,Y_val), axes = load_training_data('./data/NACE_20230610/Fig1_data/NAC_5-1_avg_pairs_BSA.npz', validation_split=0.15, verbose=True)

c = axes_dict(axes)['C']
n_channel_in, n_channel_out = X.shape[c], Y.shape[c]

plt.figure(figsize=(12,5))
plot_some(X_val[:5],Y_val[:5])
plt.suptitle('5 example validation patches (top row: source, bottom row: target)');

"""
Before we construct the actual model, 
we have to define its configuration via a Config object, which includes:
    
1. parameters of the underlying neural network,
2. the learning rate,
3. the number of parameter updates per epoch,
4. the loss function, and
5. whether the model is probabilistic or not.

The defaults should be sensible in many cases, 
so a change should only be necessary if the training process fails.
"""

config = Config(axes, n_channel_in, n_channel_out, probabilistic=False, 
                unet_residual = True, unet_n_depth=2, unet_n_first=32,
                train_learning_rate = 0.00001,
                train_epochs = 1000,
                train_steps_per_epoch=400)
print(config)
vars(config)

# We now create a DLSRS model with the chosen configuration:
model = CARE(config, 'CARE_NAC_5_1_pair_BSA', basedir='models')

# Train the DLSRS model
history = model.train(X,Y, validation_data=(X_val,Y_val))


#Plot final training history
print(sorted(list(history.history.keys())))
plt.figure(figsize=(16,5))
plot_history(history,['loss','val_loss'],['mse','val_mse','mae','val_mae']);


# Example results for validation images
plt.figure(figsize=(12,7))
_P = model.keras_model.predict(X_val[0:5])
if config.probabilistic:
    _P = _P[...,:(_P.shape[-1]//2)]
plot_some(X_val[0:5],Y_val[0:5],_P,pmax=99.5)
plt.suptitle('5 example validation patches\n'      
             'top row: input (source),  '          
             'middle row: target (ground truth),  '
             'bottom row: predicted from source');
             
# Export trained model for prediction in the future
model.export_TF()