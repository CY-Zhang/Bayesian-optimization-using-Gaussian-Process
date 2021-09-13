from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.insert(1, 'C:/Users/ASUser/Downloads/Bayesian-optimization-using-Gaussian-Process/')
# GP related libaries
saveResultsQ = False
from modules.bayes_optimization import BayesOpt, negUCB, negExpImprove
from modules.OnlineGP import OGP
# Standard python libraries
import numpy as np
import importlib
import os
import tensorflow as tf
import h5py
from datetime import datetime

saveResult = True
setBestSeen = False

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

path = 'C:/Users/ASUser/Downloads/'
acquisition_delay = 0
nrep = 1

# Iteration boundary for Nionswift-sim, in the order of C10, C12.x/y, C21.x/y, C23.x/y, C30, C32.x/y, C34.x/y
# Could be removed in real instrument
iter_bounds = [(-5e-7, 5e-7),(-5e-7, 5e-7),(-5e-7, 5e-7),(-5e-6, 5e-6),(-5e-6, 5e-6),(-3.5e-6, 3.5e-6),(-3.5e-6, 3.5e-6),(-5e-5, 5e-5),(-5e-5, 5e-5),(-3.5e-5, 3.5e-5),
(-3.5e-5, 3.5e-5),(-3.5e-5, 3.5e-5)] 
abr_activate = [True, False, True, False, False, False, False, False, False, False, False, False]
# randomize starting point, in the simulator, the global minimum is at zero point.
ndim = sum(abr_activate)
dev_ids =  [str(x + 1) for x in np.arange(ndim)] #creat device ids (just numbers)
rs = np.random.RandomState()

# for _ in range(nrep):
start_point = [[rs.rand() * 0.5 + 0.25 for x in np.arange(sum(abr_activate))]]
start_point = [[0.3, 0.3, 0.3]]
start_point = [[0.3, 0.3]]

#creat machine interface
mi_module = importlib.import_module('machine_interfaces.machine_interface_Nion')
mi = mi_module.machine_interface(dev_ids = dev_ids, start_point = start_point, CNNoption = 1, 
CNNpath = 'C:/Users/ASUser/Downloads/Bayesian-optimization-using-Gaussian-Process/CNNmodels/VGG16_nion_2ndOrder_45mradEmit+defocus_45mradApt.h5', act_list = abr_activate,
readDefault = True, detectCenter = True, exposure_t = 500, remove_buffer = False)
mi.aperture = 0

# Check the readout from machine interface
print("Initial state of the machine interface: \n")
print(mi.x)
temp = mi.getState()
print(temp[1][0][0])

# Set up GP parameters
gp_ls = [0.175, 0.302, 0.075 , 0.143, 0.164, 0.101, 0.100, 0.150, 0.288, 0.185, 0.175, 0.181] 
gp_ls = np.array([gp_ls[i] for i in np.arange(len(abr_activate)) if abr_activate[i]])
gp_ls = gp_ls / 2
print(gp_ls)
gp_amp = 0.143 / 2
gp_noise = 0.000053

gp_precisionmat =  np.array(np.diag(1/(gp_ls**2)))
ndim = len(dev_ids)
hyperparams = {'precisionMatrix': gp_precisionmat, 'amplitude_covar': gp_amp, 'noise_variance': gp_noise} 
gp = OGP(ndim, hyperparams)

#create the bayesian optimizer that will use the gp as the model to optimize the machine 
opt = BayesOpt(gp, mi, acq_func="UCB", start_dev_vals = mi.x, dev_ids = dev_ids, iter_bound= True)
opt.ucb_params = np.array([4, None])
opt.searchBoundScaleFactor = 1
# opt.ucb_params = np.array([0.02, 0.4])
opt.bounds = [(0,1) for i in np.arange(sum(abr_activate))]
print(opt.bounds)
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

if setBestSeen:
  # set corrector to best seen state and stop the camera
  mi.setX([opt.best_seen()[0]])
  print("parameter set to " + str(opt.best_seen()) + '\n')

mi.stopAcquisition()

if saveResult:
# Save the GP process files
  now = datetime.now()
  current_time = now.strftime("%H%M")
  filename = current_time + 'GPrun_' + str(ndim) + 'pars_' +  str(Niter) + 'iter_' + str(idx) + '.h5'

  f = h5py.File(path + filename, 'a')
  # Group to save the data
  grp = f.create_group('Data')
  grp.create_dataset('ronchigram', data = np.array(ronch_list))
  grp.create_dataset('prediction', data = np.array(obj_list))
  grp.create_dataset('parameters', data = np.array(status_list))

  # Group to save GP parameters
  grp2 = grp.create_group('GP_parameters')
  grp2['prmean'] = 0
  grp2['niter'] = Niter
  grp2['ndim'] = ndim
  grp2['GP_lengthscale'] = gp_ls
  grp2['GP_amp'] = gp_amp
  grp2['GP_noise'] = gp_noise
  grp2['start_point'] = start_point[0]
  grp2['ucb_param'] = str(opt.ucb_params)
  grp2['searchBoundScaleFactor'] = opt.searchBoundScaleFactor

  # Group to save the instrument parameters
  grp3 = grp.create_group('exp_parameters')
  grp3['exposure_time_ms'] = mi.ronchigram.get_current_frame_parameters()['exposure_ms']
  grp3['binning'] = mi.ronchigram.get_current_frame_parameters()['binning']
  grp3['processing'] = str(mi.ronchigram.get_current_frame_parameters()['processing'])
  grp3['active_aberration'] = abr_activate
  grp3['aberration_lim'] = mi.abr_lim

  f.close()

  print('Results saved to ' + path + filename + '\n')