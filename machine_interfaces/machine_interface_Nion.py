# Basic libraries
import numpy as np
import os
import threading
# CNN related libraries
from keras import applications
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
import tensorflow as tf
# Nion instrument related libraries
from nion.utils import Registry

class machine_interface:

    def __init__(self, dev_ids, start_point = None, CNNoption = 1, CNNpath = '', act_list = [], readDefault = False, detectCenter = False):
        # Basic setups
        os.environ["CUDA_VISIBLE_DEVICES"]="0" # specify which GPU to use
        self.pvs = np.array(dev_ids)
        self.name = 'Nion'
        self.CNNoption = CNNoption
        
        # initialize aberration list, this has to come before setting aberrations
        self.abr_list = ["C10", "C12.x", "C12.y", "C21.x", "C21.y", "C23.x", "C23.y", "C30", 
        "C32.x", "C32.y", "C34.x", "C34.y"]
        self.default = [2e-9, 2e-9, 2e-9, 20e-9, 20e-9, 20e-9, 20e-9, 0.5e-6, 0.5e-6, 0.5e-6, 0.5e-6, 0.5e-6]
        # self.abr_lim = [2e-6, 1.5e-6, 1.5e-6, 3e-5, 3e-5, 1e-5, 1e-5, 3e-4, 2e-4, 2e-4, 1.5e-4, 1.5e-4]
        self.abr_lim = [2e-7, 1.5e-7, 1.5e-7, 3e-6, 3e-6, 1e-5, 1e-5, 3e-4, 2e-4, 2e-4, 1.5e-4, 1.5e-4]
        self.activate = act_list

        # option to read existing default value, can be used when running experiment
        self.readDefault = readDefault
        self.aperture = 0

        # Initialize stem controller
        self.stem_controller = Registry.get_component("stem_controller")
        for i in range(len(self.abr_list)):
            abr_coeff = self.abr_list[i]
            _, val= self.stem_controller.TryGetVal(abr_coeff)
            if self.readDefault:
                self.default[i] = val
            print(abr_coeff + ' successfully loaded.')
        
        # Connect to ronchigram camera and setup camera parameters
        self.ronchigram = self.stem_controller.ronchigram_camera
        frame_parameters = self.ronchigram.get_current_frame_parameters()
        frame_parameters["binning"] = 1
        frame_parameters["exposure_ms"] = 250 # TODO, change to a variable

        # Acquire a test frame to set the crop region based on center detected using COM.
        # TODO: besides the center position, also detect the side length to use.
        temp = np.asarray(self.ronchigram.grab_next_to_start()[0])
        if detectCenter:
            x = np.linspace(0, temp.shape[1], num = temp.shape[1])
            y = np.linspace(0, temp.shape[0], num = temp.shape[0])
            xv, yv = np.meshgrid(x, y)
            self.center_x = int(np.average(xv, weights = temp))
            self.center_y = int(np.avergae(yv, weights = temp))
        else:
            self.center_x = temp.shape[0] / 2
            self.center_y = temp.shape[1] / 2
        
        # Allocate empty array to save the frame acquired from camera
        self.size = 128
        self.frame = np.zeros([self.size, self.size])

        # Load the CNN model for objective prediction
        if CNNoption == 1:
            # load CNN architecture in a separate thread
            threading.Thread(target = self.loadCNN(CNNpath))
            # self.CNNmodel = self.loadCNN(CNNpath); # hard coded model path for now
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
              try:
                for gpu in gpus:
                  tf.config.experimental.set_memory_growth(gpu, True)
              except RuntimeError as e:
                print(e)

        if type(start_point) == type(None):
            current_x = np.zeros(len(self.pvs)) #replace with expression that reads current ctrl pv values (x) from machine
            self.setX(current_x)
        else: 
            self.setX(start_point)

    # initialize a VGG16 model and load pre-trained weights.
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
        self.CNNmodel = new_model
        print('CNN model loaded with weights.')
        return

    # function to scale Ronchigram to [0,1]
    def scale_range(self, input, min, max):
        input += -(np.min(input))
        input /= np.max(input) / (max - min)
        input += min
        return input

    # function to scale Ronchigram to between [min, max] with the aperture considered, only rescale the part within the aperture.
    # 08-24-21, not working well based on linescan tests.
    def scale_range_aperture(input, min, max):
        hist, bin_edges = np.histogram(np.ndarray.flatten(input), bins = 'auto')
        idx = np.argmin(abs(np.gradient(hist)[0:len(hist)//2]))
        threshold = bin_edges[idx]
        input += -threshold
        input[input<0] = 0
        input /= np.max(input) / (max - min)
        input += min
        return input

    # function to generate an aperture mask on the Ronchigram
    def aperture_generator(self, px_size, simdim, ap_size):
        x = np.linspace(-simdim, simdim, px_size)
        y = np.linspace(-simdim, simdim, px_size)
        xv, yv = np.meshgrid(x, y)
        apt_mask = np.sqrt(xv*xv + yv*yv) < ap_size # aperture mask
        return apt_mask

    # set the values of activated aberration coefficients.
    def setX(self, x_new):
        self.x = x_new
        idx = 0
        idx_activate = 0
        # set activated aberration coeff to desired value, and default values for the rest
        for abr_coeff in self.abr_list:
            if self.activate[idx]:
                val = x_new[0][idx_activate] * self.abr_lim[idx] - self.abr_lim[idx] / 2    
                idx_activate += 1
            else:
                val = self.default[idx]
            # print(abr_coeff, val)
            self.stem_controller.SetVal(abr_coeff, val)
            idx += 1
        return
    
    # function to resume to default aberrations
    def resume_default(self):
        self.setX([self.default])
    
    # function to acquire a single frame by calling grab_next_to_start
    def acquire_frame(self):
        self.frame = np.zeros([self.size, self.size])
        # self.ronchigram.start_playing()
        # print('Acquiring frame')
        temp = np.asarray(self.ronchigram.grab_next_to_start()[0])
        temp = temp[self.center_y - 384 : self.center_y + 384, self.center_x - 384, self.center_x + 384]
        new_shape = [self.size, self.size]
        shape = (new_shape[0], temp.shape[0] // new_shape[0],new_shape[1], temp.shape[1] // new_shape[1])
        temp = temp.reshape(shape).mean(-1).mean(1)
        self.frame = temp
        # print('Frame acquired.')
        return

    # function to set objective based on CNN prediction, no return value, self.objective state
    # will be updated
    def getCNNprdiction(self, frame_array):
        x_list = []
        frame_array = self.scale_range(frame_array, 0, 1)
        if self.aperture != 0:
            frame_array = frame_array * self.aperture_generator(128, 40, self.aperture)
        new_channel = np.zeros(frame_array.shape)
        img_stack = np.dstack((frame_array, new_channel, new_channel))
        x_list.append(img_stack)
        x_list = np.concatenate([arr[np.newaxis] for arr in x_list])
        prediction = self.CNNmodel.predict(x_list, batch_size = 1)
        self.objective_state = 1 - prediction[0][0]
        return

    # function to collect frame and call CNN to predict objective.
    def getState(self): 
        acquire_thread = threading.Thread(target = self.acquire_frame())
        acquire_thread.start()

        # Get emittance from CNN model using the image acquired from Ronchigram camera
        if self.CNNoption == 1:
            # print('Using CNN prediction.')
            threading.Thread(target = self.getCNNprdiction(self.frame))

        # For debug purpose, cannot run without CNN in this case.
        if self.CNNoption == 0:
            print('Running without CNN.')
            self.objective_state = 1

        return np.array(self.x, ndmin = 2), np.array([[self.objective_state]])

    # function to stop acquisition on the Ronchigram camera
    def stopAcquisition(self):
        if self.ronchigram:
            self.ronchigram.stop_playing()
        return