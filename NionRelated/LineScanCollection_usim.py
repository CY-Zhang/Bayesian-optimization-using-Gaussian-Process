'''
06-17-21
Script to collect image stack with varying aberration coefficient.
Currently tested on simulator nionswift-sim.
Should be good to run on the real instrument, not calling nion-swift in the script.
'''

from nion.utils import Registry
import numpy as np
import threading

class linescan:

    def __init__(self, path):
        ### Initialize linescan runner
        # define the path to save the file
        self.path = path
        # camera parameters
        self.exposure_ms = 50
        self.binning = 1     # full frame has 2048 px.
        self.rep = 1
        self.abr_list = ["C10", "C12.x", "C12.y", "C21.x", "C21.y", "C23.x", "C23.y", "C30", 
        "C32.x", "C32.y", "C34.x", "C34.y"]
        self.default = [2e-9, 2e-9, 2e-9, 20e-9, 20e-9, 20e-9, 20e-9, 0.5e-6, 1e-6, 1e-6, 1e-6, 1e-6]
        self.default = [-11.08e-9, -16.4e-9, 3.9e-9, 344.4e-9, 224.7e-9, 20e-9, 20e-9, 0.5e-6, 0.5e-6, 0.5e-6, 0.5e-6, 0.5e-6]

    def acquire_series(self, abr_coeff, abr_range, nsteps):
        # name of aberration coefficient to vary
        self.abr_coeff = abr_coeff
        # total range of aberration in m
        self.abr_range = abr_range
        # number of steps to change the aberration coefficients.
        self.nsteps = nsteps
        default_val = self.default[self.abr_list.index(abr_coeff)]

        # initialize list for aberration and image.
        value_list = [(i - self.nsteps//2) * self.abr_range / self.nsteps + default_val for i in range(self.nsteps)]
        self.image_stack = []
        # Connect to stem controller to setup aberration
        stem_controller = Registry.get_component("stem_controller")
        success, _ = stem_controller.TryGetVal(abr_coeff)
        print(success)
        ronchigram = stem_controller.ronchigram_camera
        # start acquisition for each aberration value in the list, in a separate thread.
        for i in value_list:
            for _ in range(self.rep):
                if stem_controller.SetVal(self.abr_coeff, i):
                    threading.Thread(target = self.acquire_frame(ronchigram)).start()
                    print(self.abr_coeff + ' ' + str(i))
        # After acquisition, set the value back to the default number.
        stem_controller.SetVal(self.abr_coeff, self.default[self.abr_list.index(self.abr_coeff)])

        # save the acquired image stack.
        image_stack_array = np.asarray(self.image_stack)
        filename = self.abr_coeff + '_' + str(abr_range) + 'm_' + str(self.nsteps) + 'steps_' + str(self.exposure_ms) + 'ms_bin' + str(self.binning) + '_repx' + str(self.rep) + 'localmin_fullframe.npy'
        print(self.path + filename)
        np.save(self.path + filename, image_stack_array)
        del image_stack_array
        return

    def acquire_frame(self, ronchigram):
        temp = ronchigram.grab_next_to_start()[0].data
        # temp = temp[512:1536, 512:1536]
        temp = self.rebin(temp, [128, 128])
        # print(temp.shape)
        self.image_stack.append(temp)
        # print(len(self.image_stack))
        return
    
    def stop_playing(self):
        stem_controller = Registry.get_component("stem_controller")
        ronchigram = stem_controller.ronchigram_camera
        ronchigram.stop_playing()
        return

    def set_default(self):
        # Connect to ronchigram camera and setup camera parameters
        stem_controller = Registry.get_component("stem_controller")
        ronchigram = stem_controller.ronchigram_camera
        frame_parameters = ronchigram.get_current_frame_parameters()
        frame_parameters["binning"] = self.binning
        frame_parameters["exposure_ms"] = self.exposure_ms
        ronchigram.start_playing(frame_parameters)
        stem_controller = Registry.get_component("stem_controller")
        # set up aberration coefficients
        idx = 0
        for abr_coeff in self.abr_list:
            stem_controller.SetVal(abr_coeff, self.default[idx])
            idx += 1

    def rebin(self, arr, new_shape):
        shape = (new_shape[0], arr.shape[0] // new_shape[0],
                new_shape[1], arr.shape[1] // new_shape[1])
        return arr.reshape(shape).mean(-1).mean(1)


obj = linescan('/home/chenyu/Desktop/Bayesian-optimization-using-Gaussian-Process/NionRelated/')
obj.exposure_ms = 50
obj.binning = 1
obj.rep = 1
obj.set_default()
# obj.acquire_series('C10', 2e-6, 100)
# obj.acquire_series('C12.x', 2e-6, 100)
# obj.acquire_series('C12.y', 2e-6, 100)
obj.acquire_series('C21.x', 3e-6, 100)
obj.acquire_series('C21.y', 3e-6, 100)
# obj.acquire_series('C23.x', 3e-5, 100)
# obj.acquire_series('C23.y', 3e-5, 100)
# obj.acquire_series('C30', 4e-4, 100)
# obj.acquire_series('C32.x', 3e-4, 100)
# obj.acquire_series('C32.y', 3e-4, 100)
# obj.acquire_series('C34.x', 3e-4, 100)
# obj.acquire_series('C34.y', 3e-4, 100)
obj.stop_playing()