from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.insert(1, '/home/chenyu/Desktop/Bayesian-optimization-using-Gaussian-Process')
# GP related libaries
saveResultsQ = False
from modules.bayes_optimization import BayesOpt, negUCB, negExpImprove
from modules.OnlineGP import OGP
# Standard python libraries
import pickle
import numpy as np
import importlib
import time
import os
import tensorflow as tf
import matplotlib.pyplot as plt
# from numba import cuda

'''
06-23-21 First version of working Nion GP, tested for 1000 iterations, won't overflow GPU memory.
Maybe next step needs to refind with aperture applied.
'''

gpus = tf.config.experimental.list_physical_devices('GPU')
os.environ["CUDA_VISIBLE_DEVICES"]="0" # specify which GPU to use

if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

path = '/home/chenyu/Desktop/Bayesian-optimization-using-Gaussian-Process/NionRelated/'
acquisition_delay = 0
nrep = 1

# Iteration boundary for Nionswift-sim, in the order of C10, C12.x/y, C21.x/y, C23.x/y, C30, C32.x/y, C34.x/y
# Could be removed in real instrument
iter_bounds = [(-5e-7, 5e-7),(-5e-7, 5e-7),(-5e-7, 5e-7),(-5e-6, 5e-6),(-5e-6, 5e-6),(-3.5e-6, 3.5e-6),(-3.5e-6, 3.5e-6),(-5e-5, 5e-5),(-5e-5, 5e-5),(-3.5e-5, 3.5e-5),
(-3.5e-5, 3.5e-5),(-3.5e-5, 3.5e-5)] 
abr_activate = [True, True, True, True, True, False, False, False, False, False, False, False]
# randomize starting point, in the simulator, the global minimum is at zero point.
ndim = sum(abr_activate)
dev_ids =  [str(x + 1) for x in np.arange(ndim)] #creat device ids (just numbers)
rs = np.random.RandomState()

for _ in range(nrep):
  start_point = [[rs.rand() * 0.1 + 0.45 for x in np.arange(sum(abr_activate))]]
  print(start_point)

  #creat machine interface
  mi_module = importlib.import_module('machine_interfaces.machine_interface_Nion')
  mi = mi_module.machine_interface(dev_ids = dev_ids, start_point = start_point, CNNoption = 1, 
  CNNpath = '/home/chenyu/Desktop/Bayesian-optimization-using-Gaussian-Process/CNNmodels/VGG16_usim_40mradEmit+defocus_noApt.h5', act_list = abr_activate)
  mi.aperture = 0

  # Check the readout from machine interface
  print(mi.x)
  temp = mi.getState()
  print(temp[1][0][0])

  # Set up GP parameters
  gp_ls = [0.120, 0.112, 0.110, 0.143, 0.164, 0.101, 0.100, 0.150, 0.288, 0.185, 0.175, 0.181] 
  gp_ls = np.array([gp_ls[i] for i in np.arange(len(abr_activate)) if abr_activate[i]])
  print(gp_ls)
  gp_amp = 0.143
  gp_noise = 0.000053

  gp_precisionmat =  np.array(np.diag(1/(gp_ls**2)))
  ndim = len(dev_ids)
  hyperparams = {'precisionMatrix': gp_precisionmat, 'amplitude_covar': gp_amp, 'noise_variance': gp_noise} 
  gp = OGP(ndim, hyperparams)

  #create the bayesian optimizer that will use the gp as the model to optimize the machine 
  opt = BayesOpt(gp, mi, acq_func="UCB", start_dev_vals = mi.x, dev_ids = dev_ids, iter_bound= True)
  opt.ucb_params = np.array([2, None])
  # opt.ucb_params = np.array([0.002, 0.4])
  # opt.bounds = [(0,1) for i in np.arange(sum(abr_activate))]
  # print(opt.bounds)
  status_list = []
  obj_list = []
  ronch_list = []

  # Start running GP:
  Niter = 100
  for i in range(Niter):
    ronch_list.append(mi.frame)
    temp = opt.OptIter()
    status_list.append(temp[0][0])
    obj_list.append(temp[1][0])
    print(i)
    print(temp[1][0])

  # set corrector to best seen state and stop the camera
  mi.setX([opt.best_seen()[0]])
  print(opt.best_seen())
  mi.getState()
  mi.stopAcquisition()

  # print(np.array(status_list))
  # print(np.array(obj_list))

  # Save the GP process files
  idx = 0
  filename = 'GPrun_' + str(Niter) + 'iter_' + str(idx) + '_abr_coeff_UCB_2_0.npy'
  print(path + filename)
  while(os.path.isfile(path + filename)):
    idx += 1
    filename = 'GPrun_' + str(Niter) + 'iter_' + str(idx) + '_abr_coeff_UCB_2_0.npy'
  
  np.save(path + filename, np.array(status_list))
  filename = 'GPrun_' + str(Niter) + 'iter_' + str(idx) +'_prediction_UCB_2_0.npy'
  print(path + filename)
  np.save(path + filename, np.array(obj_list))

  filename = 'GPrun_' + str(Niter) + 'iter_' + str(idx) +'_ronchigram_UCB_2_0.npy'
  print(path + filename)
  np.save(path + filename, np.array(ronch_list))