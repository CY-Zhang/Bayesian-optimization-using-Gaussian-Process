import numpy as np
import sys
sys.path.insert(1, '/home/chenyu/Desktop/Bayesian-optimization-using-Gaussian_Process/GPTrelated/')
from uscope_calc import sim
import matplotlib.pyplot as plt
import os
import time
from modules.bayes_optimization import BayesOpt, negUCB, negExpImprove
from modules.OnlineGP import OGP
import importlib
# CNN related libraries
from keras import applications, optimizers, callbacks
from keras.models import Sequential, load_model
from keras.layers import Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
import tensorflow as tf

class machine_interface:
    def __init__(self, dev_ids, start_point = None, CNNoption = 0, CNNpath = '', DefocusOption = 0, S2 = 0.5):
        os.environ["CUDA_VISIBLE_DEVICES"]="0" # specify which GPU to use
        self.pvs = np.array(dev_ids)
        self.name = 'GPT' #name your machine interface. doesn't matter what you call it as long as it isn't 'MultinormalInterface'.
        self.CNNoption = CNNoption
        self.DefocusOption = DefocusOption
        self.S2 = S2 # the normalized objective lens value
        self.aperture = 0

        if type(start_point) == type(None):
            current_x = np.zeros(len(self.pvs)) #replace with expression that reads current ctrl pv values (x) from machine
            self.setX(current_x)
        else: 
            self.setX(start_point)

        if CNNoption == 1:
            self.CNNmodel = self.loadCNN(CNNpath); # hard coded model path for now
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
              try:
                for gpu in gpus:
                  tf.config.experimental.set_memory_growth(gpu, True)
              except RuntimeError as e:
                print(e)

        if DefocusOption == 1:
        	self.DefocusModel = self.loadCNN('CNNmodels/VGG16_defocus_test14.h5')

    def loadCNN(self, path):
        os.environ["CUDA_VISIBLE_DEVICES"]="0"
        model = applications.VGG16(weights=None, include_top=False, input_shape=(128, 128, 3))
        print('Model loaded')
        top_model = Sequential()
        top_model.add(Flatten(input_shape=model.output_shape[1:]))
        top_model.add(Dense(256, activation='relu'))
        top_model.add(Dropout(0.0))
        top_model.add(Dense(1,activation=None))
        new_model = Sequential()

        for l in model.layers:
            new_model.add(l)

        new_model.add(top_model)
        new_model.load_weights(path)
        return new_model

    def scale_range(self, input, min, max):
        input += -(np.min(input))
        input /= np.max(input) / (max - min)
        input += min
        return input

    def aperture_generator(self, px_size, simdim, ap_size):
        x = np.linspace(-simdim, simdim, px_size)
        y = np.linspace(-simdim, simdim, px_size)
        xv, yv = np.meshgrid(x, y)
        apt_mask = mask = np.sqrt(xv*xv + yv*yv) < ap_size # aperture mask
        return apt_mask

    # function that update objective lens current without restarting the GP
    def setS2(self, S2_new):
        self.S2 = S2_new

    def setX(self, x_new):
        self.x = np.array(x_new, ndmin=1)
        # add expressions to set machine ctrl pvs to the position called self.x -- Note: self.x is a 2-dimensional array of shape (1, ndim). To get the values as a 1d-array, use self.x[0]

    def CorrectDefocus(self, lens, obj):

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
        Niter = 5  # run 10 iterations for each case
        
        for i in range(Niter):
            Obj_state_s.append(opt.best_seen()[1])
            opt.OptIter()
            
        # the optimized objective lens current and corresponding defocus is saved in opt.best_seen()[0] and [1]
        res = opt.best_seen()
        del mi, gp, opt
        
        return res
    

    def getDefocus(self):
        ASCIIFILE = '/home/chenyu/Desktop/Bayesian-optimization-using-Gaussian_Process/outscope.txt'
        PNGFILE = '/home/chenyu/Desktop/Bayesian-optimization-using-Gaussian-Process/ronchigram.npy'
        os.environ["CUDA_VISIBLE_DEVICES"]="0"
        MConHBAR  =  2.59e12
        maxsig = 1  # determine how many standard deviations are we going to plot

        x_list = []
        # normalize then divided by 2 to match the contrast of Matlab simulated Ronchigrams
        # frame = self.scale_range(shadow, 0, 1) / 2 * self.aperture_generator(128, 40, 40)
        frame = np.load(PNGFILE)
        frame = self.scale_range(frame, 0, 1)
        new_channel = np.zeros(frame.shape)
        img_stack = np.dstack((frame, new_channel, new_channel))
        x_list.append(img_stack)
        x_list = np.concatenate([arr[np.newaxis] for arr in x_list])
        prediction = self.DefocusModel.predict(x_list, batch_size = 1)
        # print(prediction)
        defocus = 1 - prediction[0][0]
        print('Estimating defocus...')
        del x_list, img_stack, frame, prediction
        return defocus


    def getState(self): 
        ASCIIFILE = '/home/chenyu/Desktop/Bayesian-optimization-using-Gaussian-Process/outscope.txt'
        PNGFILE = '/home/chenyu/Desktop/Bayesian-optimization-using-Gaussian-Process/ronchigram.npy'
        # os.environ["CUDA_VISIBLE_DEVICES"]="0"
        MConHBAR  =  2.59e12
        maxsig = 1  # determine how many standard deviations are we going to plot

        x_low = np.asarray([1000, -40, 387000, -685000, -3.7515e6, 119000, 640000])
        x_high = np.asarray([2800, 40, 393000, -622500, -3.7495e6, 120300, 651000])

        xlim, ylim, shadow = sim(
                alpha = 1.0e-4*5,

                # Setup for full GPT run
                # S1 = 2.5e5,
                # S2 = 2.44e5 + self.S2 * 0.06e5,
                # H1 = self.x[0][0] * (x_high[0] - x_low[0]) + x_low[0],
                # H2 = self.x[0][0] * (x_high[0] - x_low[0]) + x_low[0] + self.x[0][1] * (x_high[1] - x_low[1]) + x_low[1],
                # S3 = self.x[0][4]* (x_high[5] - x_low[5]) + x_low[5],  # 119931.5,
                # S4 = self.x[0][5]* (x_high[6] - x_low[6]) + x_low[6],  # 648691.415,
                # S6 = self.x[0][2]* (x_high[2] - x_low[2]) + x_low[2],  # 390000,
                # S7 = self.x[0][3]* (x_high[3] - x_low[3]) + x_low[3],  # -654100.0
                # Obj = -3.7505e6,

                # # Setup for testing using single variable and change H1, H2 simultaneously.
                # H1 = self.x[0][0] * (x_high[0] - x_low[0]) + x_low[0],
                # H2 = self.x[0][0] * (x_high[0] - x_low[0]) + x_low[0],
                # S1 = 2.5e5,
                # S2 = 2.5e5,
                # S3 = 119931.5,
                # S4 = 648691.415,
                # S6 = 390000,
                # S7 = -654100.0,
                # Obj = -3.7505e6,


                # Setup for testing using varying only H1 and H2
                H1 = self.x[0][0] * (x_high[0] - x_low[0]) + x_low[0],
                H2 = self.x[0][0] * (x_high[0] - x_low[0]) + x_low[0] + self.x[0][1] * (x_high[1] - x_low[1]) + x_low[1],
                S1 = 2.5e5,
                S2 = 2.5e5,
                S3 = 119931.5,
                S4 = 648691.415,
                S6 = 390000,
                S7 = -654100.0,
                Obj = -3.7505e6, # new objective lens setting with high conv angle
             )      # the parameters that are not given an value here would be set to the default values, which could be found in uscope.py
                    # the sim function would return the Ronchigram, and save the outscope.txt file to the path that was calling this function
                    # i.e. the path of the Jupyte Notebook


        # Get emittance from CNN model using the shadow returned by GPT
        if self.CNNoption == 1:
            x_list = []
            if self.aperture != 0:
                frame = self.scale_range(shadow, 0, 1) * self.aperture_generator(128, 40, self.aperture)
            else:
                frame = self.scale_range(shadow, 0, 1)
            new_channel = np.zeros(frame.shape)
            img_stack = np.dstack((frame, new_channel, new_channel))
            x_list.append(img_stack)
            x_list = np.concatenate([arr[np.newaxis] for arr in x_list])
            prediction = self.CNNmodel.predict(x_list, batch_size = 1)
            # print(prediction)
            objective_state = 1 - prediction[0][0]
            # print('Using CNN prediction.')
            del x_list, img_stack, frame, prediction

        # # Get emittance from electron profiles in the GPT output
        # # check whether outscope file is ready in the path defined above
        if self.CNNoption == 0:
            if ~os.path.exists(ASCIIFILE):
                time.sleep(1)
            # time.sleep(10)

            # process the simulated results from outscope.txt, then remove the file
            screen =  np.loadtxt(ASCIIFILE, skiprows=5)
            
            x  = screen[:,0]
            y  = screen[:,1]
            x = x * 1e12
            y = y * 1e12  # x and y in unit of pm

            ax = np.divide(screen[:,4], screen[:,6])
            ay = np.divide(screen[:,5], screen[:,6])
            arx = np.sqrt(ax**2 + ay**2)
            index = np.where(arx < 0.04)

            x = x[index]
            y = y[index]
            ax = ax[index]
            ay = ay[index]

            # directly calculate emittance from defination for all the simulated electrons
            emit_1 = np.average(x**2 + y**2)
            emit_2 = np.average(ax**2 + ay**2)
            emit_3 = np.average(x*ax + y*ay)
            emit = np.sqrt(emit_1 * emit_2 - emit_3**2) # emittance in unit of [pm*rad]

            # return objective state as the negative sum of emittance
            # negative sum of emit is used as the BO will maximize the objective state, as a result of using the negative UCB as acquisition func
            objective_state = 1-emit
            print('Using GPT output.')
        # # print(objective_state)

        # The rest is the same for two different emittance calculation methods
        # print('saving ronchigram...')
        np.save('ronchigram.npy', shadow)
        # # save Ronchigram figure as a reference of tuning
        # # fig = plt.figure()
        # # plt.imshow(shadow)
        # # plt.savefig('ronchigram.png')
        # # os.remove(ASCIIFILE)

        return np.array(self.x, ndmin = 2), np.array([[objective_state]])
    
    
