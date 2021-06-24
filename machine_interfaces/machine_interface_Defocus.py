import numpy as np
import sys
sys.path.insert(1, '/home/chenyu/Desktop/Bayesian-optimization-using-Gaussian_Process/GPTrelated/')
from uscope_calc import sim
import matplotlib.pyplot as plt
import os
import time
# CNN related libraries
from keras import applications, optimizers, callbacks
from keras.models import Sequential, load_model
from keras.layers import Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
import tensorflow as tf

class machine_interface:
    def __init__(self, dev_ids, start_point = None, CNNpath = '', lens = []):
        os.environ["CUDA_VISIBLE_DEVICES"]="0" # specify which GPU to use
        self.pvs = np.array(dev_ids)
        self.name = 'Defocus'
        # load the current for the rest lenses, in the order of H1, dH, S6, S7, S3, S4, all are normalzied values
        self.lens = lens

        if type(start_point) == type(None):
            current_x = np.zeros(len(self.pvs)) #replace with expression that reads current ctrl pv values (x) from machine
            self.setX(current_x)
        else: 
            self.setX(start_point)
        # replace with CNN path later
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

    def setLens(self, lens_new):
        self.lens = np.array(lens_new, ndmin=1)

    def setX(self, x_new):
        self.x = np.array(x_new, ndmin=1)
        # add expressions to set machine ctrl pvs to the position called self.x -- Note: self.x is a 2-dimensional array of shape (1, ndim). To get the values as a 1d-array, use self.x[0]

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
        os.environ["CUDA_VISIBLE_DEVICES"]="0"
        MConHBAR  =  2.59e12
        maxsig = 1  # determine how many standard deviations are we going to plot

        # Same high and low range, the 4th element for defocus is not used.
        x_low = np.asarray([1000, -40, 387000, -685000, -3.7515e6, 119000, 640000])
        x_high = np.asarray([2800, 40, 393000, -622500, -3.7495e6, 120300, 651000])

        xlim, ylim, shadow = sim(
                alpha = 1.0e-4*5,
                S1 = 2.5e5,
                S2 = 2.44e5 + self.x[0][0] * 0.06e5,
                H1 = self.lens[0][0] * (x_high[0] - x_low[0]) + x_low[0],
                H2 = self.lens[0][0] * (x_high[0] - x_low[0]) + x_low[0] + self.lens[0][1] * (x_high[1] - x_low[1]) + x_low[1],
                S3 = self.lens[0][4]* (x_high[5] - x_low[5]) + x_low[5],  #119931.5,
                S4 = self.lens[0][5]* (x_high[6] - x_low[6]) + x_low[6],  #648691.415,
                S6 = self.lens[0][2]* (x_high[2] - x_low[2]) + x_low[2],  #390000,
                S7 = self.lens[0][3]* (x_high[3] - x_low[3]) + x_low[3],  #-654100.0
                Obj = -3.7505e6,

                # Option 2: control S7 to change defocus
                # alpha = 1.0e-4*5,
                # S1 = 2.5e5,
                # S2 = 2.5e5,
                # H1 = self.lens[0][0] * (x_high[0] - x_low[0]) + x_low[0],
                # H2 = self.lens[0][0] * (x_high[0] - x_low[0]) + x_low[0] + self.lens[0][1] * (x_high[1] - x_low[1]) + x_low[1],
                # S3 = self.lens[0][4]* (x_high[5] - x_low[5]) + x_low[5],  #119931.5,
                # S4 = self.lens[0][5]* (x_high[6] - x_low[6]) + x_low[6],  #648691.415,
                # S6 = self.lens[0][2]* (x_high[2] - x_low[2]) + x_low[2],  #390000,
                # S7 = self.x[0][0]* (x_high[3] - x_low[3]) + x_low[3],  #-654100.0
                # Obj = -3.7505e6,

             )      # the parameters that are not given an value here would be set to the default values, which could be found in uscope.py
                    # the sim function would return the Ronchigram, and save the outscope.txt file to the path that was calling this function
                    # i.e. the path of the Jupyte Notebook


        # Get defocus from CNN model using the shadow returned by GPT
        x_list = []
        # normalize then divided by 2 to match the contrast of Matlab simulated Ronchigrams
        # frame = self.scale_range(shadow, 0, 1) / 2 * self.aperture_generator(128, 40, 40)
        frame = self.scale_range(shadow, 0, 1)
        new_channel = np.zeros(frame.shape)
        img_stack = np.dstack((frame, new_channel, new_channel))
        x_list.append(img_stack)
        x_list = np.concatenate([arr[np.newaxis] for arr in x_list])
        prediction = self.DefocusModel.predict(x_list, batch_size = 1)
        # print(prediction)
        objective_state = 1 - prediction[0][0]
        print('Predicting defocus...')
        del x_list, img_stack, frame, prediction

        # The rest is the same for two different emittance calculation methods
        print('saving ronchigram...')
        np.save('ronchigram.npy', shadow)

        return np.array(self.x, ndmin = 2), np.array([[objective_state]])
    
    
