from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from modules.bayes_optimization import BayesOpt, negUCB, negExpImprove
from modules.OnlineGP import OGP
import numpy as np
import importlib
import sys
import os


# function that create a new defocus GP and correct the objective lens current
def FocusCorrection(lens, obj):

    ndim = 1
    dev_ids =  [str(x+1) for x in np.arange(ndim)]

    start_point = [[obj]]  
    mi_module = importlib.import_module('machine_interfaces.machine_interface_Defocus')
    mi = mi_module.machine_interface(dev_ids = dev_ids, start_point = start_point, lens = lens)
    mi.getState()
    
    gp_ls = np.array(np.ones(ndim)) * [0.317] 
    gp_amp = 0.256
    gp_noise = 0.000253
    gp_precisionmat =  np.array(np.diag(1/(gp_ls**2)))
    hyperparams = {'precisionMatrix': gp_precisionmat, 'amplitude_covar': gp_amp, 'noise_variance': gp_noise} 
    gp = OGP(ndim, hyperparams)
    
    opt = BayesOpt(gp, mi, acq_func="UCB", start_dev_vals = mi.x, dev_ids = dev_ids)
    opt.ucb_params = np.array([2, None])
    
    Obj_state_s=[]  # initialize empty Obj_state_s for each start point
    Niter = 10  # run 10 iterations for each case
    
    for i in range(Niter):
        Obj_state_s.append(opt.best_seen()[1])
        opt.OptIter()
        
    # the optimized objective lens current and corresponding defocus is saved in opt.best_seen()[0] and [1]
    res = opt.best_seen()
    del mi, gp, opt
    
    return res