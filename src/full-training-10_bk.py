#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
import matplotlib.pylab as plt

from modules.utils import load_cifar10, load_cifar100
from modules.cnn_with_spectral_pooling import CNN_Spectral_Pool


# # INSTRUCTIONS - PLEASE READ CAREFULLY
# 
# ### GPU Required
# 
# The code will only run on a machine with a GPU since it processes images channels-first.
# 
# ### Unzipping model Required
# 
# You need to run 
# 
# ```tar -zxvf best_model_10.tar.gz```
# 
# from the src/ directory before proceeding.

# In[2]:


print('Please make sure you are running on a machine with a GPU and have already unzipped the best_model_10.tar.gz file before proceeding.')


# # Download CIFAR data, if necessary, and load it

# In[3]:


xtrain, ytrain, xtest, ytest = load_cifar10(num_batches=5, get_test_data=True, channels_last=False)


# In[4]:


xtrain.shape, ytrain.shape, xtest.shape, ytest.shape


# # Best result from manual tuning

# In[5]:

'''
tf.reset_default_graph()
cnn = CNN_Spectral_Pool(M=6,
                        num_output=10,
                        verbose=True,
                        gamma=0.79)
cnn.calc_test_accuracy(xtest, ytest, 'best_cifar_10/test_1513443174.5236883')
'''

# ## Only run the rest of the notebook to retrain from scratch

# In[4]:


from modules.cnn_with_spectral_pooling import CNN_Spectral_Pool
validation_size = 4096

cnn = CNN_Spectral_Pool(M=7,
                        num_output=10,
                        verbose=True,
                        learning_rate=1.09e-3,
                        l2_norm=2.7e-5,
                        lr_reduction_factor=0.5,
                        lr_reduction_epochs=[11,21,31,41],
                        gamma=0.6336)
cnn.calc_test_accuracy(xtest, ytest, 'best_cifar_10/optimal_hyperparams_1550116437.5241544')                       

# In[5]:


tf.reset_default_graph()
cnn.train(xtrain[:-validation_size],
          ytrain[:-validation_size],
          xtrain[-validation_size:],
          ytrain[-validation_size:],
          batch_size=256,
          epochs=180,
          extra_conv_layer=True,
          use_global_averaging=True,
          model_name='optimal_hyperparams'
)
full_model_name = cnn.full_model_name


# # Calculate accuracy on test set
