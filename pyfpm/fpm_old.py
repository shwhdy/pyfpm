"""
Generic FPM solver developed by Kristina and David as a course project

Kristina Monakhova  monakhova@berkeley.edu
David Ren           david.ren@berkeley.edu

May 10, 2017

"""

from abc import ABCMeta, abstractmethod
from copy import deepcopy
import sys
import os
import numpy as np
import numpy.linalg as la
import time

# import pyfftw
import glob
import scipy.io as sio
import json
# from . import ransac
import matplotlib.pyplot as plt
from itertools import compress  # For logical indexing of lists
from libwallerlab.utilities import io as iotools
from libwallerlab.utilities import display as displaytools
import libwallerlab.optics as opticstools
# from libwallerlab.algorithms import iteralg as algorithms

class FpmOptions():
    '''
    This class contains all options for FPM processing. It should be created first with default values, then modified for each reconstruction as needed.
    '''

    def __init__(self, from_dict=None):
        self.calib_debug = True                                 # Self-Calibration Debugging Flag
        self.main_debug = True                                  # FPM debugging flag

        #  Algorithm options
        self.algorithm = "seq_gd"                               # The FPM Algorithm to use
        self.lm_delta1 = 1                                      # LMS algorithm parameter
        self.lm_delta2 = 1                                      # LMS algorithm parameter
        self.alg_nest_alpha = 0.1e-2                            # Nesterov algorithm parameter
        self.alg_nest_beta = 0.0041                             # Nesterov algorithm parameter
        self.alg_gd_step_size = 1e-2                            # Gradient Descent algorithm parameter
        self.max_it = 20                                        # Maximum number of iterations to run
        self.max_na = 1.0                                       # Maximum NA to use
        self.live_plot = False                                  # Whether to use a graphical iteration plot (currently testing)
        self.quiet = False                                      # Turns off printing
        self.live_plot_aspect = "wide"                          # Aspect ratio of live plot figure, can be "wide" or "square"
        self.live_plot_figure_size = (10, 3)                    # Figure size of live plot
        self.roi = iotools.Roi()                                # Object ROI to process
        self.solve_for_color_object = False                     # Whether to treat the object as r/g/b or not
        self.measurement_type = "amplitude"                     # Type of measurement to use. Can be "amplitude" or "intensity"
        self.objective_function_type = "LeastSquares"           # Type of objective function to use. Can be "LeastSquares" or "PoissonLikelihood"
        self.pupil_update = False                               # Whether to correct the pupil
        self.obj_init = None                                    # Object to initilize, if any

        # Auto-calibration Options
        self.led_auto_calib_enabled = False                     # Flag for enabling/disabling led auto-calibration
        self.led_auto_calib_scan_range = 1                      # Range to scan for "full" or "grad" based method
        self.led_auto_calib_scan_mode = "all"                   # Which leds to scan, can be "bf", "df", or "all"
        self.led_auto_calib_rad_pen = 0                         # Radial penalty to enforce
        self.led_auto_calib_itr_range = range(self.max_it)      # Iteration to start led auto-calibration
        self.led_auto_calib_add_error_na = 0                    # Error to add to the led positions, used for testing ONLY
        self.led_auto_calib_rigid_trans = True                 # Whether to enforce a rigid transformation at the end of each iteration
        self.led_auto_calib_rigid_trans_type = "homography"          # The type of rigid transformation - can be homog, affine, or ransac (lstsq is for testing)
        self.led_auto_calib_rigid_trans_every_led = False       # This flag enables per-led homography calibration, which basically calculates the affine transoform over and over as each LED is added. Still testing, not sure if it's useful.
        self.led_auto_calib_use_pre = False                     # Whether to use pre-calibraiton, if available
        self.led_auto_calib_use_pre_rigid = False               # Whether to enforce a ridid linear transformation on the pre-calibration data

        # Load keys from dictionary if provided
        if from_dict is not None:
            # Load fields from opts_dict into structure
            for key_name in from_dict:
                if hasattr(self, key_name):  # Check to see if attribute is in metadata
                    setattr(self, key_name, from_dict[key_name])

    def __str__(self):
        '''
        Over-ride string serialization for printing
        '''
        return(iotools.objToString(self, text_color=iotools.Color.BLUE))

class ColorCameraFilter():

    def __init__(self):
        self._valid_forward   = {"pco"     : self._pcoForward, \
                                 "optimos" : self._optimosForward}
        self._valid_backward  = {"pco"     : self._pcoBackward, \
                                 "optimos" : self._optimosBackward}

        self.pco_cali_matrix     = 1.0e1 * np.asarray([[1.0, 0.1, 0.1],\
                                                       [0.3, 1.0, 0.3],\
                                                       [0.3, 1.0, 0.3],\
                                                       [0.1, 0.1, 1.0]])
        self.optimos_cali_matrix = np.asarray([[  8484.45221429,  55634.2545    ,  29847.48467857],
                                               [  4914.77960714,  24062.43346429,  28663.16453571],
                                               [  8387.4105    ,  12739.49228571,  3109.74296429],
                                               [  9103.79560714,  55262.27217857,  28615.758     ]], dtype = 'float64')

    def filter(self, x, camera_name):
        assert camera_name in self._valid_forward, "cannot find calibration matrix for this camera!"
        return self._valid_forward[camera_name](x)

    def filter_adj(self, x, camera_name):
        assert camera_name in self._valid_backward, "cannot find calibration matrix for this camera!"
        return self._valid_backward[camera_name](x)

    def _pcoForward(self, x):
        assert x.shape[0] == 3, "need 3 channels to perform filtering on pco camera!"
        shape       = (x.shape[1]//2, 2, x.shape[2]//2, 2)
        x_color     = []

        for color_index in range(3):
            x_bin   = x[color_index, :, :].reshape(shape).mean(axis = -1).mean(axis = 1)
            x_color.append(x_bin)

        x_color     = np.reshape(x_color,(3,-1))
        x_pco_color = np.dot(self.pco_cali_matrix, x_color)
        shape       = (x.shape[1]//2, x.shape[2]//2, 2, 2)

        return x_pco_color.T.reshape(shape).transpose(0,2,1,3).reshape(x.shape[1], x.shape[2])

    def _pcoBackward(self, v):
        shape               = (v.shape[0]//2, 2, v.shape[1]//2, 2)
        d_pco_color         = v.reshape(shape).transpose(0,2,1,3).reshape(-1, 4).T
        d_pco_color_dx      = np.dot(self.pco_cali_matrix.T, d_pco_color)
        d_pco_color_dx_2    = np.empty((3, v.shape[0], v.shape[1]), dtype = 'complex128')

        shape               = (v.shape[0]//2, v.shape[1]//2)
        for color_index in range(3):
            d_pco_color_dx_2[color_index, :, :] = np.repeat(\
                                                  np.repeat(d_pco_color_dx[color_index, :].reshape(shape), 2, axis = 0),\
                                                  2, axis = 1)
        return 0.25 * d_pco_color_dx_2

    def _optimosForward(self, x):
        assert x.shape[0] == 3, "need 3 channels to perform filtering on optimos camera!"
        shape       = (x.shape[1]//2, 2, x.shape[2]//2, 2)
        x_color     = []

        for color_index in range(3):
            x_bin   = x[color_index, :, :].reshape(shape).mean(axis = -1).mean(axis = 1)
            x_color.append(x_bin)

        x_color     = np.reshape(x_color,(3,-1))
        x_optimos_color = np.dot(self.optimos_cali_matrix, x_color)
        shape       = (x.shape[1]//2, x.shape[2]//2, 2, 2)

        return x_optimos_color.T.reshape(shape).transpose(0,2,1,3).reshape(x.shape[1], x.shape[2])

    def _optimosBackward(self, v):
        shape               = (v.shape[0]//2, 2, v.shape[1]//2, 2)
        d_optimos_color         = v.reshape(shape).transpose(0,2,1,3).reshape(-1, 4).T
        d_optimos_color_dx      = np.dot(self.optimos_cali_matrix.T, d_optimos_color)
        d_optimos_color_dx_2    = np.empty((3, v.shape[0], v.shape[1]), dtype = 'complex128')

        shape               = (v.shape[0]//2, v.shape[1]//2)
        for color_index in range(3):
            d_optimos_color_dx_2[color_index, :, :] = np.repeat(\
                                                  np.repeat(d_optimos_color_dx[color_index, :].reshape(shape), 2, axis = 0),\
                                                  2, axis = 1)
        return 0.25 * d_optimos_color_dx_2

class ObjectiveFunction():

    def __init__(self):
        self._valid_forward  = {"LeastSquares"     : self._LSForward, \
                                "PoissonLikelihood": self._PoissonForward}
        self._valid_backward = {"LeastSquares"     : self._LSBackward, \
                                "PoissonLikelihood": self._PoissonBackward}
    def objFunc(self, x, func_name, data, funcVal_only = False):

        assert func_name in self._valid_forward, "cost function not implemented!"
        assert x.shape == data.shape, "shape of x and shape of data are different!"

        function_value, cache_forward = self._valid_forward[func_name](x, data)

        if funcVal_only:
            return function_value
        else:
            backprop_vector = self._valid_backward[func_name](cache_forward)
            return function_value, backprop_vector

    def _LSForward(self, x, data):
        residual = x - data
        return la.norm(residual.ravel()) ** 2, residual

    def _LSBackward(self, cache_forward):
        return cache_forward

    def _PoissonForward(self, x, data):
        return (x - data * np.log(x)).sum(), (x, data)

    def _PoissonBackward(self, cache_forward):
        x, data          = cache_forward
        dfunc_dx         = 1.0 - data / (x + 1e-16)
        dfunc_dx[x == 0] = 0.0
        return dfunc_dx


class FpmSolver():
    '''
    Main solver class for FPM
    '''
    def __init__(self, dataset, options):

        # Create an empty dataset object if one is not defined
        self.dataset = dataset

        # Create an empty FpmOptions object with default values if one is not defined
        self.options = options

        # Create a ObjectiveFunction object
        self.cost_obj = ObjectiveFunction()

        # Create a ColorCameraFilter object
        if self.options.solve_for_color_object:
            self.color_filter_obj = ColorCameraFilter()

        # Check to be sure solver is supported
        if self.options.algorithm not in {"global_gd", "global_nesterov", "global_lbfgs", "global_newton", "seq_gd", "seq_nesterov", "seq_lbfgs", "seq_newton", "seq_lma_approx", "seq_lma"}:
            raise ValueError("fpm_algorithm %s is not supported." % self.options.algorithm)

        # Load led positions using this priority level -> source_list_na_df_calib, source_list_na_bf_calib, source_list_na_init, source_list_na
        if self.options.led_auto_calib_use_pre:
            if dataset.metadata.illumination.state_list.calibrated is not None:
                if not self.options.quiet:
                    print("Using pre-calibration.")
                self.source_list_na = np.asarray(dataset.metadata.illumination.state_list.calibrated)
            elif dataset.metadata.illumination.state_list.design is not None:
                if not self.options.quiet:
                    print("Could not find pre-calibrated positions - using design positions.")
                self.source_list_na = np.asarray(dataset.metadata.illumination.state_list.design)
            else:
                raise ValueError("Could not load LED positions from dataset object!")
        else:
            self.source_list_na = np.asarray(dataset.metadata.illumination.state_list.design)

        # Set design positions and initial positions
        self.source_list_na_initial = self.source_list_na
        self.source_list_na_design = np.asarray(dataset.metadata.illumination.state_list.design)
        self.frame_state_list = dataset.frame_state_list

        if dataset.frame_list is not None:
            self.frame_list = dataset.frame_list.astype(np.float)  # put in MATLAB form for now. TODO: Make frame_list order pythonic
            if self.options.roi is not None:
                self.frame_list = self.frame_list[:, self.options.roi.y_start:self.options.roi.y_end,
                                                  self.options.roi.x_start:self.options.roi.x_end]
        else:
            raise ValueError("Dataset does not contain any data!")

        # Error checking
        print(self.frame_list.shape)
        assert self.frame_list.shape[1] > 0, 'Image size is in y (first dimension).'
        assert self.frame_list.shape[2] > 0, 'Image size is in x (second dimension).'
        assert self.dataset.metadata.illumination.spectrum.center is not None, "Illumination wavelengths are missing from metadata."
        assert self.dataset.metadata.camera.pixel_size_um is not None, "Camera pixel size missing from metadata."
        assert self.dataset.metadata.objective.na is not None, "Objective NA missing from metadata."
        assert self.dataset.metadata.objective.mag is not None, "Objective Mag missing from metadata."
        assert len(self.dataset.frame_state_list) == self.dataset.frame_list.shape[0], "Frame state list does not have same size as image stack."

        # Map of PCB board indicies, default considers all LEDs on same board
        if dataset.metadata.illumination.state_list.grouping is not None:
            self.source_list_board_idx = np.asarray(dataset.metadata.illumination.state_list.grouping)
        else:
            self.source_list_board_idx = np.zeros(len(dataset.metadata.illumination.state_list.design))

        # Configure which colors are enabled in this device
        self.colors_used = []
        for key in self.dataset.metadata.illumination.spectrum.center:
            self.colors_used.append(key)

        if self.options.solve_for_color_object:
            self.object_color_channel_count = len(self.colors_used)
        else:
            self.object_color_channel_count = 1

        self.frame_mask = np.zeros(self.frame_list.shape[0], dtype=np.bool)
        self.frame_max_na = np.zeros(self.frame_list.shape[0], dtype=np.float)
        self.led_used_mask = []

        # Determine which leds are used using frame_state_list
        for frame_index, frame in enumerate(self.dataset.frame_state_list):
            frame_na_list = []
            for time_point in frame['illumination']['sequence']:
                for led in time_point:

                    # Append absolute na to frame_na_list for comparison
                    frame_na_list.append(np.sqrt(self.source_list_na[led["index"], 0] ** 2 + self.source_list_na[led["index"], 1] ** 2))

                    # Keep track of which leds are used
                    if led["index"] not in self.led_used_mask:
                        self.led_used_mask.append(led['index'])

                    # Ensure all colors in spectrum component are in this state
                    for color_name in self.colors_used:
                        assert color_name in led['value'], "color " + color_name + " could not be found in frame:" + str(frame)

            self.frame_max_na[frame_index] = max(frame_na_list)
            if self.frame_max_na[frame_index] < self.options.max_na:
                self.frame_mask[frame_index] = True

        # Crop frame_list and frame_state_list using frame_mask
        self.frame_list = self.frame_list[self.frame_mask, :, :]
        self.frame_state_list = [i for (i, v) in zip(self.frame_state_list, self.frame_mask.tolist()) if v]
        assert len(self.frame_state_list) == self.frame_list.shape[0], "Frame state list has length of %d, frame_state_list has length %d" % (len(self.frame_state_list), self.frame_list.shape[0])

        # Update pixel size in case user has changed magnificaiton
        dataset.metadata.system.eff_pixel_size_um = dataset.metadata.camera.pixel_size_um / (dataset.metadata.objective.mag * dataset.metadata.system.mag)

        if dataset.metadata.camera.is_color:
            dataset.metadata.system.eff_pixel_size_um *= 2

        self.eff_pixel_size = dataset.metadata.system.eff_pixel_size_um

        # Image dimensions
        self.illum_na_max = min(np.max(np.sqrt(self.source_list_na[:, 0] ** 2 + self.source_list_na[:, 1] ** 2)), self.options.max_na)
        self.scale = max(np.ceil((dataset.metadata.objective.na + self.illum_na_max) / dataset.metadata.objective.na), 2).astype(np.int)
        self.recon_pixel_size = self.eff_pixel_size / self.scale
        self.m_crop = self.frame_list.shape[1]
        self.n_crop = self.frame_list.shape[2]
        self.wavelength_um = dataset.metadata.illumination.spectrum.center  # This should be a dict with color values matching the values in frame_state_list
        self.min_wavelength_um = min(self.wavelength_um.values())           # Min wavelength (used for sampling calculations)
        [self.M, self.N] = np.array([self.m_crop, self.n_crop]) * self.scale

        self.n_frames = self.frame_list.shape[0]

        if not self.options.quiet:
            print(displaytools.Color.BOLD + displaytools.Color.YELLOW + 'Initialized dataset: ' + dataset.metadata.file_header + displaytools.Color.END)
            print("    Using %d of %d images in dataset with size (%d, %d)" % (self.n_frames, len(self.dataset.frame_state_list), self.frame_list.shape[1], self.frame_list.shape[2]))

            # Print sampling parameters of imaging system
            print("    Imaging NA is %.2f (%.2fum min. feature size), Nyquist NA is %.2f " % (dataset.metadata.objective.na, 1. / (dataset.metadata.objective.na / min([self.min_wavelength_um])), min([self.min_wavelength_um]) / (2 * dataset.metadata.camera.pixel_size_um / dataset.metadata.objective.mag)))

            # Print imaging and reconstruction NA
            print("    Illumination NA is %.2f, reconstructed NA is %.2f (%.2fum min feature size)" % (self.illum_na_max, dataset.metadata.objective.na + self.illum_na_max,
            1. / ((dataset.metadata.objective.na + self.illum_na_max) / min([self.min_wavelength_um]))))

            # Print resolution parameters
            print("    Reconstructed object has %dx smaller pixels." %
                  self.scale)

        # TODO - make these work.
        # self.fourier_small = opticstools.Fourier([self.m_crop, self.n_crop],(0, 1))
        # self.fourier = opticstools.Fourier([self.M, self.N], (0, 1))

        # Define FFTW parameters (transparent to user)
        self._fftw_arr_small = pyfftw.zeros_aligned([self.m_crop, self.n_crop], 'complex128')
        self._fftw_arr_large = pyfftw.zeros_aligned([self.M, self.N], 'complex128')
        self._plan_small_f = pyfftw.FFTW(self._fftw_arr_small,
                                         self._fftw_arr_small, axes=[0, 1])
        self._plan_small_b = pyfftw.FFTW(self._fftw_arr_small,
                                         self._fftw_arr_small, axes=[0, 1],
                                         direction="FFTW_BACKWARD")
        self._plan_large_f = pyfftw.FFTW(self._fftw_arr_large,
                                         self._fftw_arr_large, axes=[0, 1])
        self._plan_large_b = pyfftw.FFTW(self._fftw_arr_large,
                                         self._fftw_arr_large, axes=[0, 1],
                                         direction="FFTW_BACKWARD")

        # Initialize reconstruction results
        self.obj = np.zeros([self.options.max_it + 1, self.M, self.N], dtype=np.complex128)
        self.cost = np.zeros(self.options.max_it + 1)
        self.pupil = np.zeros([self.m_crop, self.n_crop], dtype=np.complex128)

        # Keep track of which iteration we are on
        self.current_itr = 0

        # Live plot flags
        self.live_plot_active = False

        # Convert source_list_na to cropx and cropy
        self.na2crop()

        # Make sure extrapolated points from Regina's method are fit to a homographic linear transformation
        if self.options.led_auto_calib_use_pre_rigid:
            if not self.options.quiet:
                print("Performing rigid fit of pre-calibration...")
            self.fitLedNaToRigidTransform(global_transformation=True)

        # Add perturbation to led positions if user indicates
        if self.options.led_auto_calib_add_error_na > 0:
            if not self.options.quiet:
                print("Adding random perturbations with magnitude %.2f NA to LED positions." % options.led_auto_calib_add_error_na)

            na_perturb = self.options.led_auto_calib_add_error_na * \
                (np.random.rand(self.source_list_na.shape[0], self.source_list_na.shape[1]) - 0.5)

            self.source_list_na = self.source_list_na.copy() + na_perturb
            self.na2crop()

        # Update source_list_na
        self.source_list_na = self.crop2na()

        # Store initial source points
        self.source_list_na_init = self.crop2na()

        # Generate a mask for brightfield images
        self.brightfield_mask = np.squeeze(np.sqrt(self.source_list_na[:, 0] ** 2 + self.source_list_na[:, 1] ** 2) < dataset.metadata.objective.na)

        # Create grid in Fourier domain
        fy = np.arange(-self.m_crop/2, self.m_crop/2) / (self.eff_pixel_size * self.m_crop)
        fx = np.arange(-self.n_crop/2, self.n_crop/2) / (self.eff_pixel_size * self.n_crop)
        [fxx, fyy] = np.meshgrid(fx, fy)

        # Pupil initialization
        r = np.sqrt(fxx ** 2 + fyy ** 2)
        self.pupil      = {}
        self.pupil_mask = {}
        for color_name in self.colors_used:
            self.pupil[color_name]      = (r < (dataset.metadata.objective.na) / self.wavelength_um[color_name]).astype(np.complex128)
            self.pupil_mask[color_name] = self.pupil[color_name].copy()


        # Object initialization
        self.objf = np.zeros([self.M, self.N, self.object_color_channel_count], dtype=np.complex128)
        for led_dict in self.crop_fourier[0]:
            led_number  = led_dict['led_number']
            led_color   = led_dict['led_color']
            led_value   = led_dict['led_value']
            roi         = led_dict['roi']
            color_index = self.colors_used.index(led_color) * self.options.solve_for_color_object  # this is zero if we are solving for omnochrome object only

            roi = iotools.Roi()
            roi.x_start = np.ceil(self.N / 2 - self.n_crop / 2).astype(int)
            # roi.x_start = np.round(self.N / 2 - self.n_crop / 2).astype(int)
            roi.x_end = roi.x_start + self.n_crop
            # roi.x_end   = np.floor(self.N / 2 + self.n_crop / 2).astype(int)
            roi.y_start = np.ceil(self.M / 2 - self.m_crop / 2).astype(int)
            # roi.y_start = np.round(self.M / 2 - self.m_crop / 2).astype(int)
            roi.y_end = roi.y_start + self.m_crop
            # roi.y_end   = np.floor(self.M / 2 + self.m_crop / 2).astype(int)
            print(roi)

            # Crop and filter object spectrum by pupil
            obj_init = (self.options.obj_init if self.options.obj_init is not None else np.sqrt(self.frame_list[0, :, :]))
            #self.objf[roi.y_start:roi.y_end, roi.x_start:roi.x_end, color_index] += self.F(np.sqrt(self.frame_list[0, :, :])) * self.pupil[led_color] * led_value
            self.objf[roi.y_start : roi.y_end, roi.x_start : roi.x_end, color_index] += self.F(obj_init) * self.pupil[led_color] * led_value
            #replace self.F(framelist) with initial object


        # Generate real-space object
        self.obj = np.zeros(self.objf.shape, dtype=self.objf.dtype)
        for color_index in range(self.objf.shape[2]):
            self.obj[:, :, color_index] = self.iF(self.objf[:, :, color_index])

    def F(self, x):
        """
        Forward Fourier transform operator
        """
        if np.array_equal([self.m_crop, self.n_crop], x.shape):
            self._plan_small_f.input_array[:] = np.fft.ifftshift(x)
            return np.fft.fftshift(self._plan_small_f()).copy()
        elif np.array_equal([self.M, self.N], x.shape):
            self._plan_large_f.input_array[:] = np.fft.ifftshift(x)
            return np.fft.fftshift(self._plan_large_f()).copy()
        else:
            raise ValueError("FFT size did not match n_crop or N!")

    def iF(self, x):
        """
        Inverse Fourier transform operator
        """
        if np.array_equal([self.m_crop, self.n_crop], x.shape):
            self._plan_small_b.input_array[:] = np.fft.ifftshift(x)
            return np.fft.fftshift(self._plan_small_b()).copy()
        elif np.array_equal([self.M, self.N], x.shape):
            self._plan_large_b.input_array[:] = np.fft.ifftshift(x)
            return np.fft.fftshift(self._plan_large_b()).copy()
        else:
            raise ValueError("FFT size did not match n_crop or N!")

    def crop2na(self):
        '''
        Function to convert current LED kx/ky cropping coordinates to NA
        '''

        # Generate a list of empty lists for each led, these will be updated by the routine below
        source_list_na_up_x = [[] for i in range(self.source_list_na.shape[0])]
        source_list_na_up_y = [[] for i in range(self.source_list_na.shape[0])]

        source_list_na = self.source_list_na.copy()

        for frame_index in range(self.n_frames):
            frame_crop_fourier = self.crop_fourier[frame_index]
            for led_dict in frame_crop_fourier:
                led_number = led_dict['led_number']
                color = led_dict['led_color']
                roi = led_dict['roi']

                # Determine NA
                na_x = (roi.x_start + self.n_crop / 2 - self.N / 2) * self.wavelength_um[color] / (self.recon_pixel_size * self.N)
                na_y = (roi.y_start + self.m_crop / 2 - self.M / 2) * self.wavelength_um[color] / (self.recon_pixel_size * self.M)

                # Append to list for this LED
                source_list_na_up_x[int(led_number)].append(na_x)
                source_list_na_up_y[int(led_number)].append(na_y)

        # Take average position and assign to source_list_na
        for led_index in range(len(source_list_na_up_x)):
            if len(source_list_na_up_x[led_index]) > 0:
                na_x = sum(source_list_na_up_x[led_index]) / len(source_list_na_up_x[led_index])
                na_y = sum(source_list_na_up_y[led_index]) / len(source_list_na_up_y[led_index])
                source_list_na[led_index, 0] = na_x
                source_list_na[led_index, 1] = na_y

        return(source_list_na)

    def na2crop(self):
        '''
        Function to convert current NA to kx/ky crop coordinates. Works for multiplexing and color imagery as well as flickering LEDs.
        '''

        # This variable has the same length as frame_state_list and cntains a list of list with all leds and colors which are turned on during this acquisition
        self.crop_fourier = []

        # Loop over frames
        for frame_index in range(self.n_frames):
            frame_state = self.frame_state_list[frame_index]
            frame_shift = []

            # Looper over time points within frame exposure (flickering LEDs)
            for time_point_index in range(len(frame_state['illumination']['sequence'])):
                # Loop over leds in pattern
                for led_index in range(len(frame_state['illumination']['sequence'][time_point_index])):
                    # Loop over colors in LEDs
                    for color_name in frame_state['illumination']['sequence'][time_point_index][led_index]['value']:

                        # Extract values
                        value = frame_state['illumination']['sequence'][time_point_index][led_index]['value'][color_name] / ((2 ** self.dataset.metadata.illumination.bit_depth) - 1)
                        led_number = frame_state['illumination']['sequence'][time_point_index][led_index]['index']

                        # Only Add this led and color if it's turned on
                        if value > 0:
                            # Add this LED to list of LEDs which are on in this frame
                            pupil_shift_x = np.round(self.source_list_na[led_number][0] / self.wavelength_um[color_name] * self.recon_pixel_size * self.N)
                            pupil_shift_y = np.round(self.source_list_na[led_number][1] / self.wavelength_um[color_name] * self.recon_pixel_size * self.M)

                            # Generate ROI
                            roi = iotools.Roi()
                            # roi.x_start = np.ceil(self.N / 2 + pupil_shift_x - self.n_crop / 2).astype(int)
                            roi.x_start = np.round(self.N / 2 + pupil_shift_x - self.n_crop / 2).astype(int)
                            roi.x_end = roi.x_start + self.n_crop
                            # roi.y_start = np.ceil(self.M / 2 + pupil_shift_y - self.m_crop / 2).astype(int)
                            roi.y_start = np.round(self.M / 2 + pupil_shift_y - self.m_crop / 2).astype(int)
                            roi.y_end = roi.y_start + self.m_crop

                            # Check that crops are within reconstruction size
                            assert roi.x_start < self.N, "cropxstart (%d) is > N (%d)" % roi.x_start
                            assert roi.x_end < self.M, "cropxend (%d) is > M (%d)" % roi.x_end
                            assert roi.y_start >= 0, "cropystart (%d) is < 0" % roi.y_start
                            assert roi.y_end >= 0, "cropyend (%d) is < 0" % roi.y_end

                            # Append to list
                            frame_shift.append({"led_number" : led_number, "led_color" : color_name, "roi" : roi, 'led_value' : value})

            # Append shifts from all colors and leds in this frame
            self.crop_fourier.append(frame_shift)

    def rotMatrix(theta):
        M = np.array([[np.cos(theta * np.pi / 180), -np.sin(theta * np.pi / 180),0], [np.sin(theta * np.pi / 180), np.cos(theta * np.pi / 180),0],[0,0,1]])
        return(M)

    def scaleMatrix(s):
        M = np.array([[1, 0, 0], [0,1,0],[0,0,1. / s]])
        return(M)

    def shiftMatrix(shift):
        M = np.array([[1, 0, shift[0]], [0, 1, shift[1]], [0, 0, 1]])
        return(M)

    def applyNaTransformation(self, rotation_deg=0., scale=1., shift_na=(0.,0.), flip_xy=False, flip_x=False, flip_y=False):

        # Shift matrix
        Sh = np.array([[1, 0, shift_na[0]], [0, 1, shift_na[1]], [0, 0, 1]])

        # Rotation Matrix
        R = np.array([[np.cos(rotation_deg * np.pi / 180), -np.sin(rotation_deg * np.pi / 180), 0],
                      [np.sin(rotation_deg * np.pi / 180), np.cos(rotation_deg * np.pi / 180), 0],
                      [0, 0, 1]])

        # Scaling Matrix
        Sc = np.array([[1, 0, 0], [0,1,0],[0,0,1. / scale]])

        # Total Matrix
        M = np.dot(np.dot(Sh,R),Sc)

        na = self.source_list_na.copy()
        na = np.append(na,np.ones([np.size(na,0),1]),1)
        na_2 = np.dot(M,na.T).T
        na_2[:,0] /= na_2[:,2]
        na_2[:,1] /= na_2[:,2]
        na_2 = na_2[:,0:2]

        # Apply flip in x/y
        if flip_xy:
            tmp = na_2.copy()
            na_2[:,0] = tmp[:,1].copy()
            na_2[:,1] = tmp[:,0].copy()

        if flip_x:
            na_2[:,0] *=-1

        if flip_y:
            na_2[:,1] *=-1

        self.source_list_na = na_2.copy()
        self.source_list_na_design = na_2.copy()

    # Function for finding average rigid transform for LED positiions
    def fitLedNaToRigidTransform(self, frames_to_process=-1, boards_to_process=-1, global_transformation=False,
                                 write_results=True, mode=""):

        if type(frames_to_process) is not list:
            frames_to_process = [frames_to_process]  # Convert to list

        if type(boards_to_process) is not list:
            boards_to_process = [boards_to_process]  # Convert to list

        if frames_to_process[0] == -1 and len(frames_to_process) == 1:
            frames_to_process = np.arange(self.n_frames)

        if boards_to_process[0] == -1 and len(boards_to_process) == 1:
            boards_to_process = range(np.min(self.source_list_board_idx.astype(np.int)), np.max(self.source_list_board_idx.astype(np.int)) + 1)

        if not global_transformation:
            board_map = self.source_list_board_idx
        else:
            board_map = np.zeros(self.source_list_board_idx.shape, dtype=np.bool)
            boards_to_process = [0]

        if mode is "":
            mode = self.options.led_auto_calib_rigid_trans_type

        self.source_list_na = self.crop2na()
        source_list_na_local = self.source_list_na.copy()

        # Loop over all boards
        for board_idx in boards_to_process:

            # Define a list of points in ideal orientation
            mask_led = np.zeros(self.source_list_board_idx.shape, dtype=np.bool)
            mask_board = np.zeros(self.source_list_board_idx.shape, dtype=np.bool)
            mask_led[frames_to_process] = True
            mask_board[board_map == board_idx] = True
            mask = mask_board & mask_led

            if np.sum(mask) > 8:  # we need the problem to be well-posed
                # The two lists of points we want to update
                na_pos_design = self.source_list_na_design[mask, :]
                na_pos_updated = self.source_list_na[mask, :]

                pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])
                unpad = lambda x: x[:, :-1]

                Y = pad(na_pos_design)
                X = pad(na_pos_updated)

                if mode == 'lstsq':
                    H, res, rank, s = np.linalg.lstsq(Y, X)  # ax=b (a,b)
                    transform = lambda x: unpad(np.dot(pad(x), H))
                    source_list_na_local[mask, :] = transform(na_pos_design)
                elif mode == 'homography':
                    fp = ransac.make_homog(na_pos_design.T)
                    tp = ransac.make_homog(na_pos_updated.T)
                    H = ransac.H_from_points(fp, tp)
                    source_list_na_local[mask, :] = (ransac.normalize(H.dot(fp)).T[:, 0:2])
                elif mode == 'affine':
                    fp = ransac.make_homog(na_pos_design.T)
                    tp = ransac.make_homog(na_pos_updated.T)
                    H = ransac.Haffine_from_points(fp, tp)
                    source_list_na_local[mask, :] = (ransac.normalize(H.dot(fp)).T[:, 0:2])
                elif mode == 'ransac':
                    fp = ransac.make_homog(na_pos_design.T)
                    tp = ransac.make_homog(na_pos_updated.T)
                    my_model = ransac.RansacModel()
                    H = ransac.H_from_ransac(fp, tp, my_model, maxiter=100, match_theshold=20, n_close=20)[0]
                    source_list_na_local[mask, :] = (ransac.normalize(H.dot(fp)).T[:, 0:2])

                if write_results:
                    self.source_list_na = source_list_na_local.copy()
                    self.na2crop()

        return source_list_na_local.copy()

    def findLedNaError(self, frames_to_process=-1, scan_range=-1, radial_penalty=-1, write_results=True, grad_iter=-1, scan_mode=""):
        # Determine LEDs to run
        if type(frames_to_process) is not list:
            frames_to_process = [frames_to_process]    # Convert to list

        # If we pass -1, process all leds.
        if frames_to_process[0] < 0:
            frames_to_process = np.arange(self.n_frames)

        # Use default scan range if we don't override
        if scan_range < 0:
            scan_range = self.options.led_auto_calib_scan_range
        scan_range_total = 2 * scan_range + 1

        # Use default grad iteration if we don't override
        if grad_iter < 0:
            grad_iter = self.options.led_auto_calib_scan_range

        if radial_penalty < 0:
            radial_penalty = self.options.led_auto_calib_rad_pen

        if scan_mode == "":
            scan_mode = self.options.led_auto_calib_scan_mode

        dc_vals = np.zeros((scan_range_total, scan_range_total))
        for frame_index in frames_to_process:
            # Determine if we're going to process this LED or not, based on settings and the NA of this LED

            is_brightfield = self.brightfield_mask[frame_index]
            dc_vals_list = []
            if (is_brightfield & ((scan_mode == "all") | (scan_mode == "bf"))
               | (not is_brightfield) & ((scan_mode == "all") | (scan_mode == "df"))):

                # Variables for global update
                dk_x_up = 0
                dk_y_up = 0

                # Initialize
                dc_vals = np.zeros((scan_range_total, scan_range_total))

                # Outer Loop, gradient steps
                for gItr in np.arange(grad_iter):
                    # Inner loop, over dk_x and dk_y
                    for dkx_i in np.arange(scan_range_total).astype(np.int):
                        for dky_i in np.arange(scan_range_total).astype(np.int):

                            dkx = (-np.floor(scan_range_total / 2) + dkx_i).astype(np.int) + dk_x_up
                            dky = (-np.floor(scan_range_total / 2) + dky_i).astype(np.int) + dk_y_up


                            dc_vals[dky_i, dkx_i] = self.Afunc(frames_to_process=frame_index, func_value_only=True, fourier_crop_offset=(dkx, dky))

                            # Incorporate radial penalty function
                            radius = np.sqrt(dkx ** 2 + dky ** 2)
                            p = radial_penalty * radius
                            dc_vals[dky_i, dkx_i] = dc_vals[dky_i, dkx_i] + p

                    # Determine mininum value in scan_range
                    mIdx = np.argmin(dc_vals)
                    (dk_y, dk_x) = np.unravel_index(np.argmin(dc_vals), dc_vals.shape)
                    dk_x = dk_x - np.floor(scan_range_total / 2)
                    dk_y = dk_y - np.floor(scan_range_total / 2)

                    dc_vals_list.append(dc_vals)

                    # If no change, end gradient update
                    if (np.abs(dk_x) + np.abs(dk_y)) == 0:
                        break

                    dk_x_up += dk_x.astype(np.int)
                    dk_y_up += dk_y.astype(np.int)

                if write_results:  # Pick best result in scan_range
                    self.crop_fourier[frame_index][0]['roi'].y_start += dk_y_up
                    self.crop_fourier[frame_index][0]['roi'].y_end += dk_y_up
                    self.crop_fourier[frame_index][0]['roi'].x_start += dk_x_up
                    self.crop_fourier[frame_index][0]['roi'].x_end += dk_x_up
                else:
                    print("dkx: %d, dky: %d" % (dk_x_up, dk_y_up))
        if write_results:
            self.crop2na()
        return(dc_vals_list)  # for debugging only, return most recent dc_vals


    def rotMatrix(theta):
        M = np.array([[np.cos(theta * np.pi / 180), -np.sin(theta * np.pi / 180),0], [np.sin(theta * np.pi / 180), np.cos(theta * np.pi / 180),0],[0,0,1]])
        return(M)

    def scaleMatrix(s):
        M = np.array([[1, 0, 0], [0,1,0],[0,0,1. / s]])
        return(M)

    def shiftMatrix(shift):
        M = np.array([[1, 0, shift[0]], [0, 1, shift[1]], [0, 0, 1]])
        return(M)

    def applyNaTransformation(self, rotation_deg=0., scale=1., shift_na=(0.,0.), flip_xy=False, flip_x=False, flip_y=False):

        # Shift matrix
        Sh = np.array([[1, 0, shift_na[0]], [0, 1, shift_na[1]], [0, 0, 1]])

        # Rotation Matrix
        R = np.array([[np.cos(rotation_deg * np.pi / 180), -np.sin(rotation_deg * np.pi / 180), 0],
                      [np.sin(rotation_deg * np.pi / 180), np.cos(rotation_deg * np.pi / 180), 0],
                      [0, 0, 1]])

        # Scaling Matrix
        Sc = np.array([[1, 0, 0], [0,1,0],[0,0,1. / scale]])

        # Total Matrix
        M = np.dot(np.dot(Sh, R), Sc)

        na = self.source_list_na.copy()
        na = np.append(na,np.ones([np.size(na,0),1]),1)
        na_2 = np.dot(M,na.T).T
        na_2[:,0] /= na_2[:,2]
        na_2[:,1] /= na_2[:,2]
        na_2 = na_2[:,0:2]

        # Apply flip in x/y
        if flip_xy:
            tmp = na_2.copy()
            na_2[:,0] = tmp[:,1].copy()
            na_2[:,1] = tmp[:,0].copy()

        if flip_x:
            na_2[:,0] *=-1

        if flip_y:
            na_2[:,1] *=-1

        self.source_list_na = na_2

    def save_figures(self, phase_axis_range=(-np.pi, np.pi)):
        dataset_name = self.dataset_name
        dataset_header = self.dataset_header
        plt.style.use('dark_background')

        plt.figure(figsize=(6, 3))
        plt.plot(self.cost)
        axes = plt.gca()
        axes.set_ylim([0, 5e9])
        plt.xlabel('Iteration')
        plt.ylabel('Cost Function')
        plt.savefig(self.dataset_name + "_" + self.dataset_header + "_conv.png", transparent=True)

        fig = plt.figure(figsize=(6, 6))
        cmap = plt.cm.gray
        norm = plt.Normalize(vmin=phase_axis_range[0], vmax=phase_axis_range[1])

        plt.clf()
        amp = np.abs(self.obj)
        plt.imsave(dataset_name + "_" + dataset_header + "_amp_min[" + str(np.round(np.min(amp))) + "]_max[" + str(np.round(np.max(amp))) + "].png", amp, cmap=plt.cm.gray)

        plt.clf()
        phase = np.angle(self.obj)
        plt.imsave(dataset_name + "_" + dataset_header + "_phase_min[" + str(np.round(np.min(phase), 2)) + "]_max[" + str(np.round(np.max(phase), 2)) + "].png", phase, cmap=plt.cm.gray)

        # LED Positions
        plt.figure(figsize=(6, 6))
        plt.scatter(self.source_list_na_design[:, 0], self.source_list_na_design[:, 1], c='w', label="Original Points", alpha=1.0, s=20, marker='o')
        plt.scatter(self.source_list_na[:, 0], self.source_list_na[:, 1], c='y', s=15, marker='x', alpha=0.0)
        plt.title("LED Positions", )
        plt.xlabel("$NA_x$")
        plt.ylabel("$NA_y$")
        ax = plt.gca()
        ax.set_xlim(xmin=-1.1, xmax=1.1)
        ax.set_ylim(ymin=-1.1, ymax=1.1)
        plt.axis('image')
        plt.legend(loc=3)
        plt.show()
        plt.savefig(dataset_name + "_" + dataset_header + "_ledInitial.png", transparent=True, bbox_inches='tight')

        # LED Positions
        plt.figure(figsize=(6, 6))
        plt.scatter(self.source_list_na[:, 0], self.source_list_na[:, 1], c='y', s=15, label="Corrected Points", marker='x', alpha=1.0)
        plt.scatter(self.source_list_na_design[:, 0], self.source_list_na_design[:, 1], c='w', label="Original Points", alpha=0.3, s=20, marker='o')

        plt.title("LED Positions", )
        plt.xlabel("$NA_x$")
        plt.ylabel("$NA_y$")
        ax = plt.gca()
        ax.set_xlim(xmin=-1.1, xmax=1.1)
        ax.set_ylim(ymin=-1.1, ymax=1.1)
        plt.axis('image')
        plt.legend(loc=3)
        plt.show()
        plt.savefig(dataset_name + "_" + dataset_header + "_ledCorrected.png", transparent=True, bbox_inches='tight')

    def plotResult(self):
        '''
        Temporary function, plots amplitude and phase in gray colormap given x
        Currently does not support any other functionality
        '''

        # if not self.live_plot_active:
        self.createLivePlot()

        self.updateLivePlot()

    def Afunc(self, frames_to_process=None, forward_only=False, func_value_only=False, fourier_crop_offset=(0, 0)):
        """
        This function computes the amplitude-based cost function of FPM
        Input:
            x: vectorized 1D complex array of Fourier spectrum of 2D object (length self.N*self.M)
        Returns:
            Ax: stack of 2D real intensity images (shape [self.n_crop, self.m_crop, self.n_img])
            fval: scalar cost function value
            gradient: vectorized 1D complex array of gradient at f(x), same length as x
        Optional parameters:
            func_value_only: boolean variable, function returns fval only if true
            forward_only: boolean variable, the function returns Ax only if true, else
                          returns both Ax and gradient
        """

        if frames_to_process is None:
            frames_to_process = range(self.n_frames)
        elif (type(frames_to_process) is not list) and type(frames_to_process) is not range and type(frames_to_process) is not np.arange:
            frames_to_process = [frames_to_process]

        color_channels  = [[] for _ in range(self.object_color_channel_count)]
        objfcrop_p_if   = [deepcopy(color_channels) for _ in range(len(frames_to_process))]
        Ax              = np.empty([len(frames_to_process), self.m_crop, self.n_crop])

        # Apply forward model (for cost as well as forward_only flag)
        for index, frame_index in enumerate(frames_to_process):
            for led_dict in self.crop_fourier[frame_index]:
                led_number  = led_dict['led_number']
                led_color   = led_dict['led_color']
                led_value   = led_dict['led_value']
                roi         = led_dict['roi']
                color_index = self.colors_used.index(led_color) * self.options.solve_for_color_object  # this is zero if we are solving for omnochrome object only

                # Crop and filter object spectrum by pupil
                objfcrop_p_if[index][color_index].append(self.iF(self.objf[(fourier_crop_offset[1] + roi.y_start):(fourier_crop_offset[1] + roi.y_end), \
                                                                           (fourier_crop_offset[0] + roi.x_start):(fourier_crop_offset[0] + roi.x_end), \
                                                                           color_index] * self.pupil[led_color]))

            if self.options.solve_for_color_object:
                for objfcrop_p_if_color in objfcrop_p_if[index]:
                    if objfcrop_p_if_color == []:
                        objfcrop_p_if_color.append(np.zeros((self.m_crop, self.n_crop)))
                intensity_color = []
                for color_index in range(self.object_color_channel_count):
                    intensity_color.append((np.abs(np.asarray(objfcrop_p_if[index][color_index])) ** 2 * led_value).sum(axis = 0))
                Ax[index, :, :] = self.color_filter_obj.filter(np.asarray(intensity_color), self.dataset.metadata.camera.device_name)
            else:
                Ax[index, :, :] = (np.abs(objfcrop_p_if[index][0]) ** 2 * led_value).sum(axis = 0)

        if self.options.measurement_type is "amplitude":
            Ax              = np.sqrt(Ax)
            fpm_measurement = np.sqrt(self.frame_list[frames_to_process, :, :])
        elif self.options.measurement_type is "intensity":
            Ax              = Ax
            fpm_measurement = self.frame_list[frames_to_process, :, :]
        else:
            raise ValueError("Invalid objective function type (%s)" % self.options.measurement_type)

        if forward_only:
            return Ax.ravel()

        # Compute Cost
        if func_value_only:
            return self.cost_obj.objFunc(Ax, self.options.objective_function_type, fpm_measurement, funcVal_only = True)
        else:
            fval, backprop_vector = self.cost_obj.objFunc(Ax, self.options.objective_function_type, fpm_measurement)

        # Compute gradient
        # Gradient could be rgb or monochrome
        if self.options.solve_for_color_object:
            gradient = np.zeros([self.M, self.N, len(self.colors_used)], dtype="complex128")
        else:
            gradient = np.zeros([self.M, self.N, 1], dtype="complex128")


            for index, frame_index in enumerate(frames_to_process):
                for led_dict in self.crop_fourier[frame_index]:
                    led_number = led_dict['led_number']
                    led_color = led_dict['led_color']
                    led_value = led_dict['led_value']
                    roi = led_dict['roi']
                    color_index = self.colors_used.index(led_color) * self.options.solve_for_color_object  # this is zero if we are solving for omnochrome object only

                    if self.options.objective_function_type is "amplitude":
                        # Crop and filter object spectrum by pupil
                        objfcrop_p = self.objf[roi.y_start:roi.y_end, roi.x_start:roi.x_end, color_index] * self.pupil[led_color]

                        # Substitute amplitude
                        objfcrop_ampsub = self.F(np.sqrt(self.frame_list[frame_index, :, :]) * np.exp(1j * np.angle(self.iF(objfcrop_p))))
            else:
                backprop_vector_color       = backprop_vector[index, :, :]
                if self.options.measurement_type is "amplitude":
                    backprop_vector_color  /= (Ax[index, :, :] + 1e-16)
                elif self.options.measurement_type is "intensity":
                    pass
                else:
                    raise ValueError("Invalid objective function type (%s)" % self.options.measurement_type)
                backprop_vector_color.shape = (1,) + backprop_vector_color.shape

            multi_index = np.zeros((self.object_color_channel_count,), dtype = 'int32')

            for led_dict in self.crop_fourier[frame_index]:
                led_number  = led_dict['led_number']
                led_color   = led_dict['led_color']
                led_value   = led_dict['led_value']
                roi         = led_dict['roi']
                color_index = self.colors_used.index(led_color) * self.options.solve_for_color_object  # this is zero if we are solving for omnochrome object only

                # back-propagation of multiplexed illumination
                backprop_vector_f         = self.F(objfcrop_p_if[index][color_index][multi_index[color_index]] * backprop_vector_color[color_index, :, :])
                multi_index[color_index] += 1
                # Compute Gradient
                gradient[roi.y_start:roi.y_end, roi.x_start:roi.x_end, color_index] += np.conj(self.pupil[led_color]) * backprop_vector_f

                if self.options.pupil_update:
                    # Compute Gradient and Hessian for Pupil update
                    objfcrop                          = self.objf[roi.y_start:roi.y_end, roi.x_start:roi.x_end, color_index]
                    objfcrop_abs                      = np.abs(objfcrop)
                    hessian_pupil[:, :, color_index] += objfcrop_abs ** 2
                    gradient_pupil[:, :, color_index]+= self.pupil_mask[led_color] * objfcrop_abs * np.conj(objfcrop) * backprop_vector_f

            if self.options.pupil_update:
                if self.options.solve_for_color_object:
                    for led_color in self.colors_used:
                        color_index            = self.colors_used.index(led_color) * self.options.solve_for_color_object
                        step_size_pupil        = 1e-5 / np.max(np.abs(self.objf[:, :, color_index].ravel()))
                        self.pupil[led_color] -= step_size_pupil / (hessian_pupil[:, :, color_index] + self.options.lm_delta1) * gradient_pupil[:, :, color_index]
                else:
                    step_size_pupil        = 1.0 / np.max(np.abs(self.objf[:, :, 0].ravel()))
                    self.pupil[led_color] -= step_size_pupil / (hessian_pupil[:, :, 0] + self.options.lm_delta1) * gradient_pupil[:, :, 0]

        # Normalize gradient for multiplexing
        gradient = gradient / float(len(frames_to_process))
        return (gradient[np.abs(gradient) > 0], np.abs(gradient) > 0), fval

    def applyHessianSeq(self, x, idx, obj):
        """
        This function applies the local Hessian operator to point x
        The Hessian operator is evaluated at current point obj
        Input:
            x: 1D complex array point to which local Hessian is applied
            obj: global object to be cropped corresponding to index idx
        Return:
            Hx: 1D complex array, same type as x
        """
        x = np.reshape(x, [self.m_crop, self.n_crop])
        Hx = x.copy()
        N = float(x.size)
        g = self.iF(self.pupil * obj[self.cropystart[idx][0]:self.cropyend[idx][0],
                                     self.cropxstart[idx][0]:self.cropxend[idx][0]])

        g_norm = np.exp(1j * np.angle(g)) ** 2
        Hx = 1/4./N * np.abs(self.pupil) ** 2 * x + \
            1/4./N * np.conjugate(self.pupil) * self.F((np.exp(1j * np.angle(g)) ** 2) * self.F(np.conjugate(self.pupil)*np.conjugate(x)))
        Hx += self.options.lm_delta2 * x
        return Hx.ravel()

    def applyHessianGlobal(self, x, idx=None):
        """
        This function applies the global Hessian operator to point x
        The Hessian operator is evaluated at current point self.objf
        Input:
            x: 1D complex array point to which global Hessian is applied
        Return:
            Hx: 1D complex array, same type as x
        """

        if idx is None:
            img_to_process = range(self.n_frames)
        else:
            img_to_process = [idx]

        x = np.reshape(x, [self.M, self.N])
        Hx = np.zeros([self.M, self.N], dtype="complex128")
        for p_img in img_to_process:
            i_img = img_to_process[p_img]
            gl = self.objcrop[i_img, :, :]
            objcrop_p = self.iF(x[self.cropystart[i_img][0]:self.cropyend[i_img][0],
                                  self.cropxstart[i_img][0]:self.cropxend[i_img][0]] * self.pupil)

            HOO = np.conj(self.pupil) * self.F(objcrop_p - 0.5 * np.sqrt(self.frame_list[i_img, :, :])/(np.abs(gl) + 1e-15) * objcrop_p)

            HOtO = 0.5 * np.conj(self.pupil) * self.F(np.sqrt(self.frame_list[i_img, :, :]) * (gl ** 2) / (np.abs(gl) ** 3 + 1e-15) * objcrop_p.conj())
            Hx[self.cropystart[i_img][0]:self.cropyend[i_img][0],
               self.cropxstart[i_img][0]:self.cropxend[i_img][0]] += HOO + HOtO
        Hx = (Hx / float(self.n_frames) + self.options.lm_delta2 * x)
        Hx.shape = (Hx.size, 1)
        return Hx

    def run(self, n_iter=None):
        """
        This function reconstructs object
        """

        # Use max iteration from options or if user supplies
        if n_iter is None:
            n_iter = self.options.max_it - self.current_itr

        # Perform an extra na2crop in case user modifies led positions
        self.na2crop()

        if self.current_itr + n_iter > self.options.max_it:
            print("Reached max iteration.")
        else:
            # Compute and print cost
            self.cost[0] = self.Afunc(func_value_only=True)

            # Plot initial
            if self.current_itr == 0:
                if not self.options.quiet:
                    if not self.options.live_plot:
                        print(displaytools.Color.BLUE + "|  Iter  |     Cost       | Elapsed time (sec) | Auto-calib norm | " + displaytools.Color.END)
                        print("|% 5d   |    %.02e    |     % 7.2f        |     % 4.2f       |" % (0, self.cost[0], 0., 0.))
                    else:
                        self.createLivePlot()

            t_start = time.time()
            if self.options.algorithm == "seq_nesterov":
                objfcrop_hist = np.zeros((self.n_frames, self.m_crop, self.n_crop), dtype="complex128")

            for iter in range(n_iter):

                if (self.current_itr in self.options.led_auto_calib_itr_range) & self.options.led_auto_calib_enabled:
                    source_list_na_prev = self.crop2na()

                # Store previous objf (for global algorithms)
                objf_prev = self.objf

                # Switch based on method
                if self.options.algorithm == "seq_gerchberg_saxton":
                    for i_img in range(self.n_frames):

                        # Self-calibration inner loop
                        if self.options.led_auto_calib_enabled and self.current_itr in self.options.led_auto_calib_itr_range:
                            self.findLedNaError(frames_to_process=i_img)
                            if self.options.led_auto_calib_rigid_trans_every_led:
                                self.fitLedNaToRigidTransform(frames_to_process=range(i_img + 1), boards_to_process=self.source_list_board_idx[i_img])

                        # Amplitude substitution
                        objfcrop = self.objf[self.cropystart[i_img][0]:self.cropyend[i_img][0],
                                             self.cropxstart[i_img][0]:self.cropxend[i_img][0]]
                        objfcrop_ampsub = self.F((np.sqrt(self.frame_list[i_img, :, :]) * np.exp(1j * np.angle(self.iF(objfcrop * self.pupil)))))

                        self.objf[self.cropystart[i_img][0]:self.cropyend[i_img][0],
                                  self.cropxstart[i_img][0]:self.cropxend[i_img][0]] -= (objfcrop - objfcrop_ampsub) * self.pupil

                elif self.options.algorithm == "seq_gd":
                    for i_img in range(self.n_frames):

                        # Self-calibration inner loop
                        if self.options.led_auto_calib_enabled and self.current_itr in self.options.led_auto_calib_itr_range:
                            self.findLedNaError(frames_to_process=i_img)
                            if self.options.led_auto_calib_rigid_trans_every_led:
                                self.fitLedNaToRigidTransform(frames_to_process = range(i_img + 1), boards_to_process=self.source_list_board_idx[i_img])

                        (gradient, gradient_mask) = self.Afunc(frames_to_process=i_img)[0]
                        self.objf[gradient_mask.reshape(self.M, self.N, self.objf.shape[2])] -= self.options.alg_gd_step_size * gradient
                elif self.options.algorithm == "seq_nesterov":
                    for i_img in range(self.n_frames):
                        objfcrop = self.objf[self.cropystart[i_img][0]:self.cropyend[i_img][0],
                                             self.cropxstart[i_img][0]:self.cropxend[i_img][0]]

                        # Self-calibration inner loop
                        if self.options.led_auto_calib_enabled and self.current_itr in self.options.led_auto_calib_itr_range:
                            self.findLedNaError(frames_to_process=i_img)
                            if self.options.led_auto_calib_rigid_trans_every_led:
                                self.fitLedNaToRigidTransform(frames_to_process=range(i_img + 1), boards_to_process=self.source_list_board_idx[i_img])

                        if self.current_itr == 0:
                            objfcrop_hist[i_img:, :] = objfcrop
                            gradient = self.Afunc(self.objf, idx=i_img)[0]
                            gradient = np.reshape(gradient, [self.m_crop, self.n_crop])
                            self.objf[self.cropystart[i_img][0]:self.cropyend[i_img][0],
                                      self.cropxstart[i_img][0]:self.cropxend[i_img][0]] -= self.options.alg_nest_alpha * gradient
                        else:
                            objfcrop_d = objfcrop - objfcrop_hist[i_img, :, :]
                            objfcrop_hist[i_img, :, :] = objfcrop
                            objf_mom = self.objf.copy()
                            objf_mom[self.cropystart[i_img][0]:self.cropyend[i_img][0],
                                     self.cropxstart[i_img][0]:self.cropxend[i_img][0]] = objfcrop + self.options.alg_nest_beta * objfcrop_d
                            gradient = self.Afunc(objf_mom, idx=i_img)[0].reshape([self.m_crop, self.n_crop])
                            self.objf[self.cropystart[i_img][0]:self.cropyend[i_img][0],
                                      self.cropxstart[i_img][0]:self.cropxend[i_img][0]] -= (self.options.alg_nest_alpha * gradient - self.options.alg_nest_beta * objfcrop_d)

                        if self.options.led_auto_calib_enabled:
                            if self.current_itr in self.options.led_auto_calib_itr_range:
                                self.findLedNaError(frames_to_process=i_img)

                elif self.options.algorithm == "seq_lma_approx":

                    for i_img in range(self.n_frames):

                        # Self-calibration inner loop
                        if self.options.led_auto_calib_enabled and self.current_itr in self.options.led_auto_calib_itr_range:
                            self.findLedNaError(frames_to_process=i_img)
                            if self.options.led_auto_calib_rigid_trans_every_led:
                                self.fitLedNaToRigidTransform(frames_to_process=range(i_img + 1), boards_to_process=self.source_list_board_idx[i_img])

                        gradient = self.Afunc(frames_to_process = i_img)[0]
                        gradient = np.reshape(gradient, [self.m_crop, self.n_crop])
                        step_size = np.abs(self.pupil) / np.max(np.abs(self.pupil.ravel()))

                        hinv_approx = 1. / (np.abs(self.pupil) ** 2 + self.options.lm_delta2)

                        self.objf[self.cropystart[i_img][0]:self.cropyend[i_img][0],
                                  self.cropxstart[i_img][0]:self.cropxend[i_img][0]] -= step_size * hinv_approx * gradient

                elif self.options.algorithm == "seq_lma":
                    for i_img in range(self.n_frames):

                        # Self-calibration inner loop
                        if self.options.led_auto_calib_enabled and self.current_itr in self.options.led_auto_calib_itr_range:
                            self.findLedNaError(frames_to_process=i_img)
                            if self.options.led_auto_calib_rigid_trans_every_led:
                                self.fitLedNaToRigidTransform(frames_to_process=range(i_img + 1), boards_to_process=self.source_list_board_idx[i_img])

                        gradient = self.Afunc(self.objf, idx=i_img)[0]
                        step_size = np.abs(self.pupil) / np.max(np.abs(np.reshape(self.pupil, -1)))

                        curr_hessian = lambda x: self.applyHessianSeq(x, i_img, self.objf)
                        descent_dir = algorithms.cg(curr_hessian, -gradient, maxIter=50)[0]
                        descent_dir = np.reshape(descent_dir, [self.m_crop, self.n_crop])
                        self.objf[self.cropystart[i_img][0]:self.cropyend[i_img][0],
                                  self.cropxstart[i_img][0]:self.cropxend[i_img][0]] += 0.5 * step_size * descent_dir

                elif self.options.algorithm == "global_gd":
                        if self.current_itr > 0:  # Running this during the first iteration doesn't make sense
                            # LED Auto-calibration
                            if self.options.led_auto_calib_enabled:
                                if self.current_itr in self.options.led_auto_calib_itr_range:
                                    source_list_na_prev = np.asarray(self.crop2na())
                                    self.findLedNaError(frames_to_process=-1)
                                    if self.options.led_auto_calib_rigid_trans:
                                        self.fitLedNaToRigidTransform()
                                    self.source_list_na = np.asarray(self.crop2na())
                                    print("Auto-calibration norm: %.02f" % np.linalg.norm(self.source_list_na - source_list_na_prev))

                                    if self.options.led_auto_calib_add_error_na > 0.0:
                                        print("Auto-calibration error is: %.02f" % np.linalg.norm(self.source_list_na - self.source_list_na_design))

                        gradient = self.Afunc(self.objf)[0]

                        x0 = self.objf.ravel()
                        x = x0 - self.options.alg_gd_step_size * np.amax(np.abs(x0)) / np.amax(np.abs(gradient)) * gradient
                        self.objf = np.reshape(x, [self.M, self.N]).copy()

                elif self.options.algorithm == "global_nesterov":
                    objf_d = self.objf - objf_prev
                    objf_prev = self.objf.copy()
                    gradient = self.Afunc((self.objf + self.options.alg_nest_beta * objf_d))[0]
                    gradient = np.reshape(gradient, [self.M, self.N])
                    self.objf -= (self.options.alg_nest_alpha * gradient - self.options.alg_nest_beta * objf_d)
                    self.obj = self.iF(self.objf) * (self.scale ** 2)
                    self.cost[self.current_itr + 1] = self.Afunc(self.objf, func_value_only=True)

                    # LED Auto-calibration
                    if self.options.led_auto_calib_enabled:
                        if self.current_itr in self.options.led_auto_calib_itr_range:
                            print("Performing Auto-calibration")
                            source_list_na_prev = self.crop2na()
                            self.findLedNaError(frames_to_process=-1)
                            print("Auto-calibration norm: %.02f" % np.linalg.norm(self.source_list_na - self.source_list_na_prev))

                elif self.options.algorithm == "global_lbfgs":
                    raise NotImplementedError("lbfgs needs to be modified to work with single-class model.")
                    # self.frame_list_it = 0
                    #
                    # def compute_l_bfgs_cost(x):
                    #
                    #     # LED Auto-calibration
                    #     if self.options.led_auto_calib_enabled:
                    #         if self.current_itr in self.options.led_auto_calib_itr_range:
                    #             print("Performing Auto-calibration")
                    #             source_list_na_prev = self.crop2na()
                    #             self.findLedNaError(frames_to_process=-1)
                    #             print("Auto-calibration norm: %.02f" % np.linalg.norm(self.source_list_na_up - self.source_list_na_prev))
                    #
                    #     self.frame_list_it += 1
                    #     self.cost[self.frame_list_it] = self.Afunc(x, func_value_only=True)
                    #     self.objf = np.reshape(x, [self.M, self.N]).copy()
                    #     self.obj = self.iF(self.objf) * (self.scale ** 2)
                    #     print("|  %02d   |  %.02e  |        %.02f        |" % (self.frame_list_it, self.cost[self.frame_list_it], time.time() - t_start))
                    # x0 = self.objf.ravel()
                    # x, f, d = algorithms.lbfgs(self.Afunc, x0, iprint=1, maxiter=self.maxit-1, disp=1, callback=compute_l_bfgs_cost)
                    # iter = np.minimum(d["nit"], self.maxit)
                    # self.cost = self.cost[0:iter + 1]
                    # self.obj = self.obj

                elif self.options.algorithm == "global_newton":
                    # LED Auto-calibration
                    if self.options.led_auto_calib_enabled:
                        if self.current_itr in self.options.led_auto_calib_itr_range:
                            print("Performing Auto-calibration")
                            source_list_na_prev = self.crop2na()
                            if self.options.led_auto_calib_rigid_trans:
                                self.findLedNaError(frames_to_process=-1)
                                self.fitLedNaToRigidTransform()
                            print("Auto-calibration norm: %.02f" % np.linalg.norm(self.source_list_na - self.source_list_na_prev))

                    gradient, fval = self.Afunc(self.objf)
                    descent_dir, info = algorithms.cgs(self.applyHessianGlobal, -gradient, maxiter=100, tol=1e-8)
                    descent_dir = descent_dir.ravel()
                    x0 = self.objf.ravel()
                    x, step_size = algorithms._linesearch(self.Afunc, x0, descent_dir, gradient, t=0.05,
                                                          stepSize0=200, gamma=0.8, funcVal_last=fval)[0:2]
                    self.objf = np.reshape(x, [self.M, self.N]).copy()
                else:
                    raise ValueError('Invalid FPM method')

                # Update LED positions if using auto-calibration
                if (self.current_itr in self.options.led_auto_calib_itr_range) & self.options.led_auto_calib_enabled:
                    self.source_list_na = self.crop2na()
                    source_grad = np.linalg.norm(self.source_list_na - source_list_na_prev)
                    if self.options.led_auto_calib_rigid_trans:
                        self.fitLedNaToRigidTransform()

                    if self.options.led_auto_calib_add_error_na > 0.0:
                        if not self.options.quiet:
                            print("Auto-calibration error is: %.02f" % np.linalg.norm(self.source_list_na - self.source_list_na_design))
                else:
                    source_grad = 0.

                # Incriment iteration count
                self.current_itr += 1

                # Compute and print cost
                for i_img in range(self.n_frames):
                    self.cost[self.current_itr] += self.Afunc(func_value_only=True, frames_to_process=i_img)

                if not self.options.quiet:
                    if not self.options.live_plot:
                        print("|% 5d   |    %.02e    |     % 7.2f        |     % 4.2f       |" % (self.current_itr, self.cost[self.current_itr], time.time() - t_start, source_grad))
                    else:
                        self.updateLivePlot()

        # Go back to real space of the object
        for color_channel_index in range(self.object_color_channel_count):
            self.obj[:, :, color_channel_index] = self.iF(self.objf[:, :, color_channel_index]) * (self.scale ** 2)

        return

    def createLivePlot(self, cmap='gray'):
        if self.options.live_plot_aspect is "wide":
            self.fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=self.options.live_plot_figure_size)
        elif self.options.live_plot_aspect is "square":
            self.fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2,  figsize=self.options.live_plot_figure_size)

        # Return object to real space
        obj = self.iF(np.squeeze(self.objf))

        # Make initial absorption plot
        self.obj_abs_plot = ax2.imshow(np.abs(obj), cmap=cmap)
        ax2.set_aspect(1.)
        ax2.set_title("Absorption")
        self.abs_cbar = plt.colorbar(self.obj_abs_plot, ax=ax2)

        # Make initial phase plot
        self.obj_phase_plot = ax3.imshow(np.angle(obj), cmap=cmap)
        ax3.set_aspect(1.)
        ax3.set_title("Phase")
        self.phase_cbar = plt.colorbar(self.obj_phase_plot, ax=ax3)

        # Make initial convergence plot
        convergence = self.cost
        self.conv_plot, = ax1.plot(convergence)
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Cost")
        ax1.set_title('Convergence')
        ax1.set(adjustable='box-forced')
        ax1.set_ylim(ymin=0, ymax=np.max(convergence))

        # Make initial source plot
        source_list_na = self.source_list_na
        source_list_na_design = self.source_list_na_design
        source_list_mask = np.asarray(self.led_used_mask)
        self.source_list_na_plot = ax4.scatter(self.source_list_na[source_list_mask, 0],
                                               self.source_list_na[source_list_mask, 1], c='y', s=15, label="Corrected Points", marker='x', alpha=1.0)
        self.source_list_na_design_plot = ax4.scatter(self.source_list_na_design[source_list_mask, 0],
                                                      self.source_list_na_design[source_list_mask, 1], c='w', label="Original Points", alpha=0.3, s=20, marker='o')

        ax4.set_title("LED Positions")
        ax4.set_xlabel("$NA_x$")
        ax4.set_ylabel("$NA_y$")

        ax_lim = 1.1 * max(np.max(np.abs(source_list_na)), np.max(np.abs(source_list_na_design)))
        ax4.set_xlim(xmin=-ax_lim, xmax=ax_lim)
        ax4.set_ylim(ymin=-ax_lim, ymax=ax_lim)
        ax4.set_aspect(1.)

        # Draw the figure
        self.fig.canvas.draw()
        self.live_plot_active = True

    def updateLivePlot(self):

        # Create live plot if it doesn't exist already
        if not self.live_plot_active:
            self.createLivePlot()

        # Update absorption plot
        obj = self.iF(np.squeeze(self.objf))
        self.obj_abs_plot.set_data(np.abs(obj))
        self.obj_abs_plot.set_clim([np.min(np.abs(obj)), np.max(np.abs(obj))])
        self.abs_cbar.set_clim(vmin=np.min(np.abs(obj)), vmax=np.max(np.abs(obj)))

        # Update phase plot
        self.obj_phase_plot.set_data(np.angle(obj))
        self.obj_phase_plot.set_clim([np.min(np.angle(obj)), np.max(np.angle(obj))])
        self.phase_cbar.set_clim(vmin=np.min(np.angle(obj)), vmax=np.max(np.angle(obj)))

        # Update convergence plot
        self.conv_plot.set_data(np.arange(self.cost[:self.current_itr].shape[0]), self.cost[:self.current_itr])
        self.conv_plot.axes.set_ylim(ymin=np.min(self.cost[:self.current_itr]), ymax=np.max(self.cost[:self.current_itr]))

        # Update source plot
        source_list_mask = np.asarray(self.led_used_mask)
        self.source_list_na_plot.set_offsets(self.source_list_na[source_list_mask,:])
        self.source_list_na_design_plot.set_offsets(self.source_list_na_design[source_list_mask,:])

        self.fig.canvas.draw()

class FpmMultiSolver():
    '''
    This class is used to solve for multiple objects simultaneously using the same options structure. It is NOT fot multiplexing! (Use FpmSolver for this)
    '''

    def __init__(self, dataset_list, options):

        # Create an empty dataset object if one is not defined
        self.dataset_list = dataset_list

        # Create an empty FpmOptions object with default values if one is not defined
        self.options = options

        # Make actual FPM solver objects
        self.solver_list = []
        for dataset in dataset_list:
            if self.options.algorithm in {"global_gd", "global_nesterov", "global_lbfgs", "global_newton"}:
                self.solver_list.append(FpmGlobal(dataset, self.options))
            elif self.options.algorithm in {"seq_gd", "seq_nesterov", "seq_lbfgs", "seq_newton", "seq_lma_approx"}:
                self.solver_list.append(FpmSequential(dataset, self.options))
            else:
                raise ValueError("fpm_algorithm %s is not supported." % self.options.fpm_algorithm)

    def run(self, n_iter=None):
        '''
        This method runs the FPM algorithm for all datasets, coupling the na solvers as desitred
        '''
        if n_iter is None:
            n_iter = 25

        # Initialize a large source_list
        source_list_na = np.zeros((500, 2))

        for itr in range(n_iter):
            source_list_na_list = []
            max_led = 0
            for fpm_solver in self.solver_list:
                fpm_solver.run(n_iter=1)
                source_list_na_list.append(fpm_solver.source_list_na.copy())
                max_led = max(max_led, np.size(source_list_na_list[-1], 0))

            for led_idx in range(max_led):
                o_ct = 0  # Number of objects which use this led

                for obj_idx in range(len(source_list_na_list)):
                    if np.size(source_list_na_list[obj_idx], 0) > led_idx:
                        o_ct += 1
                        source_list_na[led_idx, :] += source_list_na_list[obj_idx][led_idx, :]

                if o_ct > 0:
                    source_list_na[led_idx, :] = source_list_na[led_idx, :] / o_ct

                # for obj_idx in range(len(self.solver_list)):
                #     if np.size(self.solver_list[obj_idx].source_list_na, 0) > led_idx:
                #         self.solver_list[obj_idx].source_list_na[led_idx, :] = source_list_na[led_idx, :]
            print("Finished iteration %d of %d" % (itr + 1, n_iter))

    def plotResult(self, index):
        self.solver_list[index].plotResult()

    def plotNaResult(self, index, range_to_plot=-1, na=None):
        '''
        Plot the LED positions in NA coordinates
        '''
        if range_to_plot is -1:
            range_to_plot = np.arange(self.solver_list[index].source_list_na_init.shape[0])

        plt.figure()
        plt.scatter(self.solver_list[index].source_list_na_design[range_to_plot, 0],
                    self.solver_list[index].source_list_na_design[range_to_plot, 1], c='w', label="source_list_na_design")

        plt.scatter(self.solver_list[index].source_list_na_init[range_to_plot, 0],
                    self.solver_list[index].source_list_na_init[range_to_plot, 1], c='b', label="source_list_na_init")

        plt.scatter(self.solver_list[index].source_list_na[range_to_plot, 0],
                    self.solver_list[index].source_list_na[range_to_plot, 1], c='r', label="source_list_na")


        plt.title("Source NA")
        plt.xlim((-1., 1.))
        plt.xlabel("NA_x")
        plt.xlabel("NA_y")
        plt.ylim((-1., 1.))
        plt.legend()
        plt.show()
