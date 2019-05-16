"""
Generic FPM solver developed by Kristina and David as a course project

Kristina Monakhova  monakhova@berkeley.edu
David Ren           david.ren@berkeley.edu

May 10, 2017

"""

from abc import ABCMeta, abstractmethod
import sys
import os
import numpy as np
import numpy.linalg as la
import time
import labalg.iteralg as algorithms
import pyfftw
import glob
import labutil.iotools
import scipy.io as sio
import json
import ransac
import matplotlib.pyplot as plt
import labutil.iotools as iotools
import labutil.displaytools as displaytools


class FpmPlot():
    '''
    This class handles plotting of an fpm object. It also allows interactive updating.
    '''
    def __init__(self, fpm_obj, cmap='gray', figsize=(12, 4), figaspect="wide"):
        if figaspect is "wide":
            self.fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=figsize)
        elif figaspect is "square":
            self.fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2,  figsize=figsize)

        obj = fpm_obj.obj

        self.obj_abs_plot = ax2.imshow(np.abs(obj), cmap=cmap)
        ax2.set_aspect(1.)
        ax2.set_title("Absorption")
        self.abs_cbar = plt.colorbar(self.obj_abs_plot, ax=ax2)

        self.obj_phase_plot = ax3.imshow(np.angle(obj), cmap=cmap)
        ax3.set_aspect(1.)
        ax3.set_title("Phase")
        self.phase_cbar = plt.colorbar(self.obj_phase_plot, ax=ax3)

        convergence = fpm_obj.cost
        self.conv_plot, = ax1.plot(convergence)
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Cost")
        ax1.set_title('Convergence')
        ax1.set(adjustable='box-forced')
        ax1.set_ylim(ymin=0, ymax=np.max(convergence))

        source_list_na = fpm_obj.source_list_na
        source_list_na_design = fpm_obj.source_list_na_design

        self.source_list_na_plot = ax4.scatter(fpm_obj.source_list_na[:, 0],
                                               fpm_obj.source_list_na[:, 1], c='y', s=15, label="Corrected Points", marker='x', alpha=1.0)
        self.source_list_na_design_plot = ax4.scatter(fpm_obj.source_list_na_design[:, 0],
                                                      fpm_obj.source_list_na_design[:, 1], c='w', label="Original Points", alpha=0.3, s=20, marker='o')

        ax4.set_title("LED Positions")
        ax4.set_xlabel("$NA_x$")
        ax4.set_ylabel("$NA_y$")

        ax_lim = 1.1 * max(np.max(np.abs(source_list_na)), np.max(np.abs(source_list_na_design)))
        ax4.set_xlim(xmin=-ax_lim, xmax=ax_lim)
        ax4.set_ylim(ymin=-ax_lim, ymax=ax_lim)
        ax4.set_aspect(1.)

        self.fig.canvas.draw()


    def update(self, fpm_obj, iteration_idx=-1):
        self.obj_abs_plot.set_data(np.abs(fpm_obj.obj))
        self.obj_abs_plot.set_clim([np.min(np.abs(fpm_obj.obj)), np.max(np.abs(fpm_obj.obj))])
        self.abs_cbar.set_clim(vmin=np.min(np.abs(fpm_obj.obj)), vmax=np.max(np.abs(fpm_obj.obj)))

        self.obj_phase_plot.set_data(np.angle(fpm_obj.obj))
        self.obj_phase_plot.set_clim([np.min(np.angle(fpm_obj.obj)), np.max(np.angle(fpm_obj.obj))])
        self.phase_cbar.set_clim(vmin=np.min(np.angle(fpm_obj.obj)), vmax=np.max(np.angle(fpm_obj.obj)))

        self.conv_plot.set_data(np.arange(fpm_obj.cost[:iteration_idx].shape[0]), fpm_obj.cost[:iteration_idx])
        self.conv_plot.axes.set_ylim(ymin=np.min(fpm_obj.cost[:iteration_idx]), ymax=np.max(fpm_obj.cost[:iteration_idx]))

        self.source_list_na_plot.set_offsets(fpm_obj.source_list_na)
        self.source_list_na_design_plot.set_offsets(fpm_obj.source_list_na_design)

        self.fig.canvas.draw()


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
        self.max_img = -1                                       # Maximum number of images to use
        self.max_na = 1.0                                       # Maximum NA to use
        self.live_plot = False                                  # Whether to use a graphical iteration plot (currently testing)
        self.quiet = False                                      # Turns off printing
        self.live_plot_aspect = "wide"                          # Aspect ratio of live plot figure, can be "wide" or "square"
        self.live_plot_figure_size = (12, 3)                     # Figure size of live plot
        self.roi = iotools.Roi()                                # Object ROI to process

        # Auto-calibration Options
        self.led_auto_calib_enabled = False                     # Flag for enabling/disabling led auto-calibration
        self.led_auto_calib_scan_range = 1                      # Range to scan for "full" or "grad" based method
        self.led_auto_calib_scan_mode = "all"                   # Which leds to scan, can be "bf", "df", or "all"
        self.led_auto_calib_rad_pen = 0                         # Radial penalty to enforce
        self.led_auto_calib_itr_range = range(self.max_it)      # Iteration to start led auto-calibration
        self.led_auto_calib_add_error_na = 0                    # Error to add to the led positions, used for testing ONLY
        self.led_auto_calib_rigid_trans = False                 # Whether to enforce a rigid transformation at the end of each iteration
        self.led_auto_calib_rigid_trans_type = "homog"          # The type of rigid transformation - can be homog, affine, or ransac (lstsq is for testing)
        self.led_auto_calib_rigid_trans_every_led = False       # This flag enables per-led homography calibration, which basically calculates the affine transoform over and over as each LED is added. Still testing, not sure if it's useful.
        self.led_auto_calib_use_pre = False                     # Whether to use pre-calibraiton, if available
        self.led_auto_calib_use_pre_rigid = False               # Whether to enforce a rigid linear transformation on the pre-calibration data

        # Multiplexing options
        self.obj_init = False                                   # Whether to initialize an object for gd.

        # REMOVE THESE
        self.color_channels_used = ['g']

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


class FpmSolver():

    def __init__(self, dataset, options):

        # Create an empty dataset object if one is not defined
        self.dataset = dataset

        # Create an empty FpmOptions object with default values if one is not defined
        self.options = options

        self.color_channels_used = self.options.color_channels_used

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

        # Set design positions as initial positions
        self.source_list_na_design = np.asarray(dataset.metadata.illumination.state_list.design)
        self.frame_state_list = dataset.frame_state_list

        if dataset.frame_list is not None:
            self.frame_list = dataset.frame_list.astype(np.float)  # put in MATLAB form for now. TODO: Make frame_list order pythonic
            if self.options.roi is not None:
                self.frame_list = self.frame_list[:, self.options.roi.y_start:self.options.roi.y_end,
                                                  self.options.roi.x_start:self.options.roi.x_end]
        else:
            raise ValueError("Dataset does not contain any data!")

        assert self.frame_list.shape[1] > 0, 'Image size is 0'
        assert self.dataset.metadata.illumination.spectrum.center_r_um is not None, "Red illumination wavelength missing from metadata"
        assert self.dataset.metadata.illumination.spectrum.center_g_um is not None, "Green illumination wavelength missing from metadata"
        assert self.dataset.metadata.illumination.spectrum.center_b_um is not None, "Blue illumination wavelength missing from metadata"
        assert self.dataset.metadata.camera.pixel_size_um is not None, "Camera pixel size missing from metadata"
        assert self.dataset.metadata.objective.na is not None, "Objective NA missing from metadata"
        assert self.dataset.metadata.objective.mag is not None, "Objective Mag missing from metadata"

        # List of color channels that can be used in illumination
        self.color_list = ['r', 'g', 'b']

        # Update pixel size in case user has changed magnification
        dataset.metadata.system.eff_pixel_size_um = dataset.metadata.camera.pixel_size_um / (dataset.metadata.objective.mag * dataset.metadata.system.mag)
        self.eff_pixel_size = dataset.metadata.system.eff_pixel_size_um

        # Order frame_list and source_list_na so the first image is the center led
        self.img_sort_indicies = np.argsort(np.sqrt(self.source_list_na[:, 0] ** 2 + self.source_list_na[:, 1] ** 2))

        # Select images to use
        self.frame_mask = np.ones(self.frame_list.shape[0], dtype=np.bool)
        if options.max_img > 0:
            self.frame_mask = (np.arange(self.frame_list.shape[0]) < options.max_img)

        if options.max_na < 1.0:
            self.frame_mask *= (np.sqrt(self.source_list_na[:, 0] ** 2 + self.source_list_na[:, 1] ** 2) <= options.max_na)

        self.n_frames = np.sum(self.frame_mask)

        # Reduce image and na coordinates to correct sizes
        # frame_mask is an array of booleans-- why are we cropping self.source_list_na by it, it's an array of boolean values
        self.frame_list = self.frame_list[self.frame_mask, :, :].copy()

        # Image dimensions
        self.illum_na_max = np.max(np.sqrt(self.source_list_na[:, 0] ** 2 + self.source_list_na[:, 1] ** 2))
        self.scale = max(np.ceil((dataset.metadata.objective.na + self.illum_na_max) / dataset.metadata.objective.na), 2).astype(np.int) + 1
        self.recon_pixel_size = self.eff_pixel_size / self.scale
        self.m_crop = self.frame_list.shape[1]
        self.n_crop = self.frame_list.shape[2]

        # Define R/G/B colors for each LED
        self.wavelength_list = [dataset.metadata.illumination.spectrum.center_r_um, dataset.metadata.illumination.spectrum.center_g_um, dataset.metadata.illumination.spectrum.center_b_um]

        [self.M, self.N] = np.array([self.m_crop, self.n_crop]) * self.scale

        if not self.options.quiet:
            print(displaytools.Color.BOLD + displaytools.Color.YELLOW + 'Initialized dataset: ' + dataset.metadata.file_header + displaytools.Color.END)
            print("    Using %d of %d images in dataset with size (%d, %d)" % (self.n_frames, self.frame_mask.shape[0], self.frame_list.shape[1], self.frame_list.shape[2]))

            # Print imaging and reconstruction NA
            print("    Imaging NA is %.2f, illumination NA is %.2f, reconstructed NA is %.2f" % (dataset.metadata.objective.na, self.illum_na_max, dataset.metadata.objective.na + self.illum_na_max))

            # Print resolution parameters
            print("    Imaging resolution is %.2f um, reconstructed resolution is %.2f um, reconstructed image has %dx smaller pixels." %
                  (1. / (dataset.metadata.objective.na / min(self.wavelength_list)), 1. / ((dataset.metadata.objective.na + self.illum_na_max) / min(self.wavelength_list)), self.scale))

            # Print sampling parameters of imaging system
            print("    Imaging NA is %.2f, Nyquist NA %.2f, " % (dataset.metadata.objective.na, min(self.wavelength_list) / (2 * dataset.metadata.camera.pixel_size_um / dataset.metadata.objective.mag)))

            # Print sampling parameters of reconstruction
            print("    Reconstructed NA is %.2f, Nyquist Reconstruction NA is %.2f" % (dataset.metadata.objective.na + self.illum_na_max, min(self.wavelength_list) / (2 * self.recon_pixel_size)))

        # Map of PCB board indicies, default considers all LEDs on same board
        # If dataset.metadata.illumination.source_list_rigid_groups is not None:
        if dataset.metadata.illumination.state_list.grouping is not None:
            # self.source_list_board_idx = np.asarray(dataset.metadata.illumination.source_list_rigid_groups)[self.frame_mask]
            self.source_list_board_idx = np.asarray(dataset.metadata.illumination.state_list.grouping)[self.frame_mask]
        else:
            self.source_list_board_idx = np.zeros(self.n_frames)

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

        # Store frame_state_list
        self.frame_state_list = dataset.frame_state_list

        # Multiplexing parameters
        # self.multiplexing = True
        # if self.multiplexing:
        #     self.numlits = np.ones(np.size(self.n_frames))
        #     crop_size = (self.n_frames, max(self.numlits), self.m_crop, self.n_crop)
        # else:
        #     self.numlits = np.zeros(self.n_frames, dtype=int)
        #     for frame_index in range(self.n_frames):
        #         self.numlits[frame_index] = 1
        #         crop_size = (self.n_frames, 1, self.m_crop, self.n_crop)

        # Convert source_list_na to cropx and cropy
        # print(self.source_list_na)
        self.na2crop()

        # Determine number of channels
        max_length = 0

        # Check that crops are within reconstruction size
        assert np.max(self.cropxend) < self.N, "cropxend (%d) is > N (%d)" % (np.max(self.cropxend), self.N)
        assert np.max(self.cropyend) < self.M, "cropyend (%d) is > M (%d)" % (np.max(self.cropyend), self.M)
        assert np.min(self.cropxstart) >= 0, "cropxstart (%d) is < 0" % (np.min(self.cropxstart))
        assert np.min(self.cropystart) >= 0, "cropystart (%d) is < 0" % (np.min(self.cropystart))

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

        # TODO figure out why this would be necessary-- basically inverses what na2crop does, but na2crop keeps
        # the original source_list_na intact, so idk why we would use this function.
        # Update source_list_na
        # self.source_list_na = self.crop2na()

        # Store initial source points
        # self.source_list_na_init = self.crop2na()

        # Generate a mask for brightfield images
        self.brightfield_mask = np.squeeze(np.sqrt(self.source_list_na[:, 0] ** 2 + self.source_list_na[:, 1] ** 2) < dataset.metadata.objective.na)

        # Create grid in Fourier domain
        fy = np.arange(-self.m_crop/2, self.m_crop/2) / (self.eff_pixel_size * self.m_crop)
        fx = np.arange(-self.n_crop/2, self.n_crop/2) / (self.eff_pixel_size * self.n_crop)
        [fxx, fyy] = np.meshgrid(fx, fy)

        # Pupil initialization
        r = (fxx ** 2 + fyy ** 2) ** 0.5
        if self.dataset.metadata.camera.is_color:
            self.pupil = []
            for wavelength in self.wavelength_list:
                self.pupil.append(r < (dataset.metadata.objective.na) / wavelength)
            self.pupil = np.asarray(self.pupil)
        else:
            self.pupil = (r < (dataset.metadata.objective.na) / self.wavelength_list[0]).astype(np.complex128)

        # Object initialization
        if self.dataset.metadata.camera.is_color:
            self.objf = np.zeros([self.M, self.N, 3], dtype=np.complex128)
        else:
            self.objf = np.zeros([self.M, self.N], dtype=np.complex128)

        if self.options.obj_init:
            self.objf[np.floor((self.M - self.m_crop) / 2).astype(np.int):np.floor((self.M + self.m_crop)/2).astype(np.int),
                      np.floor((self.N - self.n_crop) / 2).astype(np.int):np.floor((self.N + self.n_crop)/2).astype(np.int)] = self.F(dataset.obj_init)
        else:
            self.objf[np.floor((self.M - self.m_crop) / 2).astype(np.int):np.floor((self.M + self.m_crop)/2).astype(np.int),
                      np.floor((self.N - self.n_crop) / 2).astype(np.int):np.floor((self.N + self.n_crop)/2).astype(np.int)] += self.F(np.sqrt(self.frame_list[0, :, :])) * self.pupil
            self.obj = self.iF(self.objf) / (self.scale ** 2)

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
        source_list_na_up = np.zeros(self.source_list_na.shape)

        for frame_index in range(len(self.cropystart)):
            for led_index in range(len(self.cropystart)[frame_index]):


        if np.ndim(self.cropxstart) is 1:
            source_list_na_up[:, 0] = (self.cropxstart + self.n_crop / 2 - self.N / 2) * self.wavelength_list / (self.recon_pixel_size * self.N)
            source_list_na_up[:, 1] = (self.cropystart + self.m_crop / 2 - self.M / 2) * self.wavelength_list / (self.recon_pixel_size * self.M)
        else:
            source_list_na_up[:, 0] = (self.cropxstart[:, 0] + self.n_crop / 2 - self.N / 2) * self.wavelength_list / (self.recon_pixel_size * self.N)
            source_list_na_up[:, 1] = (self.cropystart[:, 0] + self.m_crop / 2 - self.M / 2) * self.wavelength_list / (self.recon_pixel_size * self.M)
        return(source_list_na_up)

    def na2crop(self):
        '''
        Function to convert current NA to kx/ky crop coordinates
        '''
        pupilshifty = []
        pupilshiftx = []
        for frame_index in range(self.n_frames):
            frame_state = self.frame_state_list[frame_index]
            frame_pupilshiftx = []
            frame_pupilshifty = []

            for led_index in range(len(frame_state['illumination']['sequence'])):
                led_pupilshiftx = [0, 0, 0]
                led_pupilshifty = [0, 0, 0]

                for color_index, color_str in enumerate(self.color_list):
                    # TODO Make this work for flickering LEDs
                    val = frame_state['illumination']['sequence'][led_index][0]['value'][color_str] / ((2 ** self.dataset.metadata.illumination.bit_depth) - 1)
                    led_number = frame_state['illumination']['sequence'][led_index][0]['index']

                    na_x = self.source_list_na[led_number][0]
                    na_y = self.source_list_na[led_number][1]

                    led_pupilshiftx[color_index] = np.round(na_x / self.wavelength_list[color_index] * self.recon_pixel_size * self.N)
                    led_pupilshifty[color_index] = np.round(na_y / self.wavelength_list[color_index] * self.recon_pixel_size * self.M)

                    # Add this led to frame pupilshift
                    frame_pupilshiftx.append(led_pupilshiftx)
                    frame_pupilshifty.append(led_pupilshifty)

            # Update global pupilshiftx
            pupilshiftx.append(frame_pupilshiftx)
            pupilshifty.append(frame_pupilshifty)

        # Cropping index in Fourier domain
        self.cropystart = (self.M / 2 + pupilshifty - self.m_crop / 2).astype(int)
        self.cropyend = (self.M / 2 + pupilshifty + self.m_crop / 2).astype(int)
        self.cropxstart = (self.N / 2 + pupilshiftx - self.n_crop / 2).astype(int)
        self.cropxend = (self.N / 2 + pupilshiftx + self.n_crop / 2).astype(int)

    #  Function for finding average rigid transform for LED positiions
    def fitLedNaToRigidTransform(self, leds_to_process=-1, boards_to_process=-1, global_transformation=False,
                                 write_results=True, mode=""):

        if type(leds_to_process) is not list:
            leds_to_process = [leds_to_process]  # Convert to list

        if type(boards_to_process) is not list:
            boards_to_process = [boards_to_process]  # Convert to list

        if leds_to_process[0] == -1 and len(leds_to_process) == 1:
            leds_to_process = np.arange(self.n_frames)

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
            mask_led[leds_to_process] = True
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
                elif mode == 'homog':
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

    def findLedNaError(self, leds_to_process=-1, scan_range=-1, radial_penalty=-1, write_results=True, grad_iter=-1, scan_mode=""):
        # Determine LEDs to run
        if type(leds_to_process) is not list:
            leds_to_process = [leds_to_process]    # Convert to list

        # If we pass -1, process all leds.
        if leds_to_process[0] < 0:
            leds_to_process = np.arange(self.n_frames)

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
        for img_idx in leds_to_process:
            # Determine if we're going to process this LED or not, based on settings and the NA of this LED

            is_brightfield = self.brightfield_mask[img_idx]
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

                            I_p = np.zeros(self.frame_list[img_idx, :, :].shape)
                            for l_idx in np.arange(np.size(self.cropystart, 1)):

                                I_p = I_p + np.abs(self.iF(self.objf[(dky + self.cropystart[img_idx][l_idx]):(dky + self.cropyend[img_idx][l_idx]),
                                                                     (dkx + self.cropxstart[img_idx][l_idx]):(dkx + self.cropxend[img_idx][l_idx])] * self.pupil)) ** 2

                            I_m = self.frame_list[img_idx, :, :]
                            mean_m = max(np.mean(I_m), 1e-10)
                            mean_p = max(np.mean(I_p), 1e-10)

                            I_m = (I_m / mean_m) - 1.0
                            I_p = (I_p / mean_p) - 1.0

                            dc_vals[dky_i, dkx_i] = np.sum(np.abs(I_m - I_p))

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

                if write_results is True:  # Pick best result in scan_range
                    self.cropystart[img_idx] += dk_y_up
                    self.cropyend[img_idx] += dk_y_up
                    self.cropxstart[img_idx] += dk_x_up
                    self.cropxend[img_idx] += dk_x_up
                else:
                    print("dkx: %d, dky: %d" % (dk_x_up, dk_y_up))
        if write_results:
            self.crop2na()
        return(dc_vals_list)  # for debugging only, return most recent dc_vals

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
        amp = np.abs(self.obj[:, :, -1])
        plt.imsave(dataset_name + "_" + dataset_header + "_amp_min[" + str(np.round(np.min(amp))) + "]_max[" + str(np.round(np.max(amp))) + "].png", amp, cmap=plt.cm.gray)

        plt.clf()
        phase = np.angle(self.obj[:, :, -1])
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

    def plotResult(self, figsize=(10, 3), use_legend=False):
        '''
        Temporary function, plots amplitude and phase in gray colormap given x
        Currently does not support any other functionality
        '''
        x = self.obj
        fig_handle = plt.figure(figsize=figsize)

        plt.subplot(141)
        cost_handle = plt.plot(self.cost)
        plt.title('Convergence')
        plt.xlabel('Iteration')
        plt.ylabel('Cost Function')
        # ax=plt.gca()
        # ax.set_aspect(1)
        # plt.axis('square')

        plt.subplot(142)
        abs_handle = plt.imshow(np.abs(x), cmap="gray")
        plt.title("Amplitude")
        plt.xlabel("x")
        plt.ylabel("y")
        # plt.axis('square')

        phase = np.angle(x)
        # phase = np.unwrap(phase, discont = 5, axis = 1)
        plt.subplot(143)
        phase_handle = plt.imshow(phase, cmap="gray")
        plt.title("Phase", )
        plt.xlabel("x")
        plt.ylabel("y")
        # plt.axis('square')

        plt.subplot(144)
        plt.scatter(self.source_list_na_design[:, 0], self.source_list_na_design[:, 1], c='w', label="Original Points", alpha=0.7, s=20)
        na_handle = plt.scatter(self.source_list_na[:, 0], self.source_list_na[:, 1], c='r', label="Corrected Positions", alpha=1, s=2)

        plt.title("LED Positions", )
        plt.xlabel("NA_x")
        plt.ylabel("NA_y")
        plt.axis('square')

        if use_legend:
            plt.legend()

        return fig_handle, cost_handle, abs_handle, phase_handle, na_handle

    def plotRawData(self):
        from matplotlib.widgets import Slider, Button, RadioButtons

        fig, ax = plt.subplots()
        plt.subplots_adjust(left=0.25, bottom=0.1)
        a0 = 0
        ax = plt.imshow(self.frame_list[a0, :, :])

        axcolor = 'lightgoldenrodyellow'
        axamp = plt.axes([0.25, 0.01, 0.65, 0.03], facecolor=axcolor)
        samp = Slider(axamp, 'Image Index', 0, self.frame_list.shape[0], valinit=a0, valfmt='%0.0f')

        def update(val):
            amp = int(np.round(samp.val))
            ax.set_data(self.frame_list[amp, :, :])
            fig.canvas.draw_idle()
        samp.on_changed(update)

        if self.cam_is_color:
            rax = plt.axes([0.025, 0.5, 0.15, 0.15], facecolor=axcolor)
            radio = RadioButtons(rax, ('red', 'blue', 'green'), active=0)

            def colorfunc(label):
                l.set_color(label)
                fig.canvas.draw_idle()
            radio.on_clicked(colorfunc)

        plt.show()

    def plotNaResult(self, range_to_plot=-1, show_led_numbers=False):
        '''
        Plot the LED positions in NA coordinates
        '''
        if range_to_plot is -1:
            range_to_plot = np.arange(self.source_list_na_init.shape[0])

        plt.figure()

        plt.scatter(self.source_list_na[range_to_plot, 0],
                    self.source_list_na[range_to_plot, 1], c='r', label="source_list_na")

        if not show_led_numbers:
            plt.scatter(self.source_list_na_design[range_to_plot, 0],
                        self.source_list_na_design[range_to_plot, 1], c='w', label="source_list_na_design")

            plt.scatter(self.source_list_na_init[range_to_plot, 0],
                        self.source_list_na_init[range_to_plot, 1], c='b', label="source_list_na_init")
        else:
            for index, led_position in enumerate(self.source_list_na):
                plt.text(led_position[0], led_position[1], str(index))

        ax = plt.gca()
        ax.set_title("LED Positions")
        ax.set_xlabel("$NA_x$")
        ax.set_ylabel("$NA_y$")

        ax_lim = 1.1 * max(max(np.max(np.abs(self.source_list_na)), np.max(np.abs(self.source_list_na_design))), np.max(np.abs(self.source_list_na_init)))
        ax.set_xlim(xmin=-ax_lim, xmax=ax_lim)
        ax.set_ylim(ymin=-ax_lim, ymax=ax_lim)
        ax.set_aspect(1.)


#     multiplexing gradient-- temporarily here for inspiration
#     def Afunc(self, x, forward_only=False, funcVal_only=False):
#         """
#         This function computes the sequential amplitude-based cost function and gradient of FPM with respect to idx
#
#         Input:
#             x: vectorized 1D complex array of Fourier spectrum of 2D object (length self.N*self.M)
#         Returns:
#             Ax: 1D real amplitude image (shape [self.n_crop, self.m_crop])
#             fval: scalar cost function value
#             gradient: vectorized 1D complex array of gradient at f(x), Ax_idx
#         Optional parameters:
#             funcVal_only: boolean variable, function returns fval only if true
#             forward_only: boolean variable, the function returns Ax only if true, else
#                           returns both Ax and gradient
#         """
#         x = np.reshape(x, [self.N, self.M])
#         Ax = np.zeros([self.n_crop, self.m_crop, self.n_frames])
#         # Register all LED intensity images
#         objcrop_all = {}
#         for frame_index in range(self.n_frames):
#             for led_index in range(self.numlits[frame_index]):
#                 cur_key = self.cropystart[frame_index][led_index] * self.N + self.cropxstart[frame_index][led_index]
#                 if cur_key not in objcrop_all:
#                     objcrop_all[cur_key] = self.iF(x[self.cropystart[frame_index][led_index]:self.cropyend[frame_index][led_index],
#                                                      self.cropxstart[frame_index][led_index]:self.cropxend[frame_index][led_index]] * self.pupil)
#         # Apply forward model
#         for frame_index in range(self.n_frames):
#             for led_index in range(self.numlits[frame_index]):
#                 cur_key = self.cropystart[frame_index][led_index] * self.N + self.cropxstart[frame_index][led_index]
#                 Ax[:, :, frame_index] += np.abs(objcrop_all.get(cur_key)) ** 2
#         Ax = np.sqrt(Ax)
#         # print la.norm(Ax ** 2)
#         if forward_only:
#             return Ax.ravel()
#         # Compute cost
#         fval = la.norm((np.sqrt(self.frame_list) - Ax).ravel()) ** 2
#
#         # Compute gradient
#         if funcVal_only:
#             return fval
#         else:
#             gradient = np.zeros([self.N, self.M], dtype="complex128")
#             for frame_index in range(self.n_frames):
#                 numlit = self.numlits[frame_index]
#                 for led_index in range(numlit):
#                     cur_key = self.cropystart[frame_index][led_index] * self.N + self.cropxstart[frame_index][led_index]
#                     objcrop = objcrop_all[cur_key]
#                     gradient[self.cropystart[frame_index][led_index]:self.cropyend[frame_index][led_index],
#                              self.cropxstart[frame_index][led_index]:self.cropxend[frame_index][led_index]] -= \
#                         np.conj(self.pupil) * (self.F(objcrop / (Ax[:, :, frame_index] + 1e-30)
#                                                       * np.sqrt(self.frame_list[frame_index, :, :])) -
#                                                x[self.cropystart[frame_index][led_index]:self.cropyend[frame_index][led_index],
#                                                  self.cropxstart[frame_index][led_index]:self.cropxend[frame_index][led_index]] * self.pupil)
#             gradient = gradient.ravel() / float(self.n_frames)
#             return gradient, fval

    def Afunc0(self, x, forward_only=False, funcVal_only=False, idx=0):
        """
        This function computes the sequential amplitude-based cost function and gradient of FPM with respect to idx

        Input:
            x: vectorized 1D complex array of Fourier spectrum of 2D object (length self.N*self.M)
        Returns:
            Ax: 1D real intensity image (shape [self.n_crop, self.m_crop])
            fval: scalar cost function value
            gradient: vectorized 1D complex array of gradient at f(x), Ax_idx
        Optional parameters:
            funcVal_only: boolean variable, function returns fval only if true
            forward_only: boolean variable, the function returns Ax only if true, else
                          returns both Ax and gradient
        """
        x = np.reshape(x, [self.M, self.N])
        Ax = np.zeros([self.m_crop, self.n_crop])

        # Apply forward model
        Ax = np.abs(self.iF(x[self.cropystart[idx][0]:self.cropyend[idx][0],
                              self.cropxstart[idx][0]:self.cropxend[idx][0]] * self.pupil))
        if forward_only:
            return Ax.ravel()

        # Compute cost
        fval = la.norm((np.sqrt(self.frame_list[idx, :, :]).ravel() - Ax.ravel())) ** 2

        # Compute gradient
        if funcVal_only:
            return fval
        else:
            gradient = np.zeros([self.m_crop, self.n_crop], dtype="complex128")
            objfcrop_p = x[self.cropystart[idx][0]:self.cropyend[idx][0],
                           self.cropxstart[idx][0]:self.cropxend[idx][0]] * self.pupil
            objfcrop_ampsub = self.F(np.sqrt(self.frame_list[idx, :, :]) * np.exp(1j * np.angle(self.iF(objfcrop_p))))
            gradient = - np.conj(self.pupil) * (objfcrop_ampsub - objfcrop_p)
            gradient = gradient.ravel()
            return gradient, fval

    def Afunc(self, frames_to_process=None, forward_only=False, funcVal_only=False):
        """
        This function computes the amplitude-based cost function of FPM
        Input:
            x: vectorized 1D complex array of Fourier spectrum of 2D object (length self.N*self.M)
        Returns:
            Ax: stack of 2D real intensity images (shape [self.n_crop, self.m_crop, self.n_img])
            fval: scalar cost function value
            gradient: vectorized 1D complex array of gradient at f(x), same length as x
        Optional parameters:
            funcVal_only: boolean variable, function returns fval only if true
            forward_only: boolean variable, the function returns Ax only if true, else
                          returns both Ax and gradient
        """

        if frames_to_process is None:
            frames_to_process = range(self.n_frames)
        elif (type(frames_to_process) is not list) and type(frames_to_process) is not range and type(frames_to_process) is not np.arange:
            frames_to_process = [frames_to_process]

        Ax = np.zeros([len(frames_to_process), self.m_crop, self.n_crop])

        # Apply forward model (for cost as well as forward_only flag)
        for index in range(len(frames_to_process)):
            frame_index = frames_to_process[index]
            for led_index in range(len(self.cropystart[frame_index])):
                for color_index, color in enumerate(self.color_list):
                    if color in self.options.color_channels_used:
                        Ax[index, :, :] += np.abs(self.iF(self.objf[self.cropystart[frame_index][led_index][color_index]:self.cropyend[frame_index][led_index][color_index],
                                                  self.cropxstart[frame_index][led_index][color_index]:self.cropxend[frame_index][led_index][color_index]] * self.pupil))
        if forward_only:
            return Ax.ravel()

        # Compute Cost
        fval = la.norm((np.sqrt(self.frame_list[frames_to_process, :, :].ravel()) - Ax.ravel())) ** 2

        # Compute gradient
        if funcVal_only:
            return fval
        else:
            gradient = np.zeros([self.M, self.N], dtype="complex128")
            for index, frame_index in enumerate(frames_to_process):
                for led_index in range(len(self.cropystart[frame_index])):
                    for color_index, color in enumerate(self.color_list):
                        if color in self.options.color_channels_used:
                            # Crop current object spectra
                            objfcrop_p = self.objf[self.cropystart[frame_index][led_index][color_index]:self.cropyend[frame_index][led_index][color_index],
                                                   self.cropxstart[frame_index][led_index][color_index]:self.cropxend[frame_index][led_index][color_index]] * self.pupil

                            objfcrop_ampsub = self.F(np.sqrt(self.frame_list[frame_index, :, :]) * np.exp(1j * np.angle(self.iF(objfcrop_p))))

                            gradient[self.cropystart[frame_index][led_index][color_index]:self.cropyend[frame_index][led_index][color_index],
                                     self.cropxstart[frame_index][led_index][color_index]:self.cropxend[frame_index][led_index][color_index]] -= np.conj(self.pupil) * (objfcrop_ampsub - objfcrop_p)

                gradient = gradient / float(len(frames_to_process))

            return gradient, fval

    def Afunc1(self, frames_to_process=None, forward_only=False, funcVal_only=False):
        """
        This function computes the amplitude-based cost function of FPM
        Input:
            x: vectorized 1D complex array of Fourier spectrum of 2D object (length self.N*self.M)
        Returns:
            Ax: stack of 2D real intensity images (shape [self.n_crop, self.m_crop, self.n_frames])
            fval: scalar cost function value
            gradient: vectorized 1D complex array of gradient at f(x), same length as x
        Optional parameters:
            funcVal_only: boolean variable, function returns fval only if true
            forward_only: boolean variable, the function returns Ax only if true, else
                          returns both Ax and gradient
        """

        if frames_to_process is None:
            frames_to_process = range(self.n_frames)
        elif (type(frames_to_process) is not list) and type(frames_to_process) is not range and type(frames_to_process) is not np.arange:
            frames_to_process = [frames_to_process]

        #  Forward model is a list of images
        Ax = np.zeros([len(frames_to_process), self.m_crop, self.n_crop])

        #  Apply forward model to images (allow multiplexing)
        for frame_index in range(len(frames_to_process)):
            for led_index in range(len(self.cropystart[frame_index])):
                Ax[frame_index, :, :] += np.abs(self.iF(self.objf[self.cropystart[frame_index][led_index][0]:self.cropyend[frame_index][led_index][0],
                                                self.cropxstart[frame_index][led_index][0]:self.cropxend[frame_index][led_index][0]] * self.pupil))

        if forward_only:
            return Ax.ravel()

        # Compute Cost
        fval = 0
        for frame_index in range(len(frames_to_process)):
            fval += la.norm(np.sqrt(self.frame_list[frame_index, :, :].ravel()) - Ax[frame_index, :, :].ravel()) ** 2

        # Compute gradient
        if funcVal_only:
            return fval
        else:

            mask = np.zeros([len(frames_to_process), self.M, self.N], dtype=np.bool)

            gradient_list = []
            mask_list = []
            gradient = np.zeros([self.M, self.N], dtype="complex128")
            for index, frame_index in enumerate(frames_to_process):
                gradient *= 0  # Set gradient to zero
                for led_index in range(len(self.cropystart[frame_index])):
                    objfcrop_p = self.objf[self.cropystart[frame_index][led_index][0]:self.cropyend[frame_index][led_index][0],
                                           self.cropxstart[frame_index][led_index][0]:self.cropxend[frame_index][led_index][0]] * self.pupil

                    objfcrop_ampsub = self.F(np.sqrt(self.frame_list[frame_index, :, :]) * np.exp(1j * np.angle(self.iF(objfcrop_p))))

                    gradient[self.cropystart[frame_index][led_index][0]:self.cropyend[frame_index][led_index][0],
                             self.cropxstart[frame_index][led_index][0]:self.cropxend[frame_index][led_index][0]] += np.conj(self.pupil) * (objfcrop_ampsub - objfcrop_p)

                mask_list.append(np.abs(gradient) > 0)
                gradient_list.append(gradient[np.abs(gradient) > 0])

        return(gradient_list, mask_list), fval

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
            frames_to_process = range(self.n_frames)
        else:
            frames_to_process = [idx]

        x = np.reshape(x, [self.M, self.N])
        Hx = np.zeros([self.M, self.N], dtype="complex128")
        for p_img in frames_to_process:
            frame_index = frames_to_process[p_img]
            gl = self.objcrop[frame_index, 0, :, :]       #added another index for led_idx
            objcrop_p = self.iF(x[self.cropystart[frame_index][0]:self.cropyend[frame_index][0],
                                  self.cropxstart[frame_index][0]:self.cropxend[frame_index][0]] * self.pupil)

            HOO = np.conj(self.pupil) * self.F(objcrop_p - 0.5 * np.sqrt(self.frame_list[frame_index, :, :])/(np.abs(gl) + 1e-15) * objcrop_p)

            HOtO = 0.5 * np.conj(self.pupil) * self.F(np.sqrt(self.frame_list[frame_index, :, :]) * (gl ** 2) / (np.abs(gl) ** 3 + 1e-15) * objcrop_p.conj())
            Hx[self.cropystart[frame_index][0]:self.cropyend[frame_index][0],
               self.cropxstart[frame_index][0]:self.cropxend[frame_index][0]] += HOO + HOtO
        Hx = (Hx / float(self.n_frames) + self.options.lm_delta2 * x)
        Hx.shape = (Hx.size, 1)
        return Hx

    def run(self, n_iter=None):
        """
        This function reconstructs object
        max_it is set very high by default so it will be ignored.
        """

        # Use max iteration from options or if user supplies
        if n_iter is None:
            n_iter = self.options.max_it - self.current_itr

        if self.current_itr + n_iter > self.options.max_it:
            print("Reached max iteration.")
        else:
            # Compute and print cost
            self.cost[0] = self.Afunc(funcVal_only=True, frames_to_process=range(self.n_frames))

            # Plot initial
            if self.current_itr == 0:
                if not self.options.quiet:
                    if not self.options.live_plot:
                        print(displaytools.Color.BLUE + "|  Iter  |     Cost       | Elapsed time (sec) | Auto-calib norm | " + displaytools.Color.END)
                        print("|% 5d   |    %.02e    |     % 7.2f        |     % 4.2f       |" % (0, self.cost[0], 0., 0.))
                    else:
                        self.plotter = FpmPlot(self, figsize = self.options.live_plot_figure_size, figaspect=self.options.live_plot_aspect)

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
                    for frame_index in range(self.n_frames):

                        # Self-calibration inner loop
                        if self.options.led_auto_calib_enabled and self.current_itr in self.options.led_auto_calib_itr_range:
                            self.findLedNaError(leds_to_process=frame_index)
                            if self.options.led_auto_calib_rigid_trans_every_led:
                                self.fitLedNaToRigidTransform(leds_to_process=range(frame_index + 1), boards_to_process=self.source_list_board_idx[frame_index])

                        # Amplitude substitution
                        objfcrop = self.objf[self.cropystart[frame_index][0]:self.cropyend[frame_index][0],
                                             self.cropxstart[frame_index][0]:self.cropxend[frame_index][0]]
                        objfcrop_ampsub = self.F((np.sqrt(self.frame_list[frame_index, :, :]) * np.exp(1j * np.angle(self.iF(objfcrop * self.pupil)))))

                        self.objf[self.cropystart[frame_index][0]:self.cropyend[frame_index][0],
                                  self.cropxstart[frame_index][0]:self.cropxend[frame_index][0]] -= (objfcrop - objfcrop_ampsub) * self.pupil

                elif self.options.algorithm == "seq_gd":
                    for frame_index in range(self.n_frames):

                        # Self-calibration inner loop
                        if self.options.led_auto_calib_enabled and self.current_itr in self.options.led_auto_calib_itr_range:
                            self.findLedNaError(leds_to_process=frame_index)
                            if self.options.led_auto_calib_rigid_trans_every_led:
                                self.fitLedNaToRigidTransform(leds_to_process=range(frame_index + 1), boards_to_process=self.source_list_board_idx[frame_index])

                        # Amplitude substitution
                        # (grad_list, mask_list), fval = self.Afunc(frames_to_process=frame_index)
                        #
                        # for index, gradient in enumerate(grad_list):
                        #     self.objf[mask_list[index]] -= self.options.alg_gd_step_size * gradient

                        gradient, fval = self.Afunc(frames_to_process=frame_index)

                        # objfcrop = self.objf[self.cropystart[i_img][0]:self.cropyend[i_img][0],
                        #                      self.cropxstart[i_img][0]:self.cropxend[i_img][0]]

                        # gradient = np.reshape(gradient, [self.m_crop, self.n_crop])
                        # self.objf[self.cropystart[frame_index][0][0]:self.cropyend[frame_index][0][0],
                        #           self.cropxstart[frame_index][0][0]:self.cropxend[frame_index][0][0]] -= self.options.alg_gd_step_size * gradient

                        self.objf -= self.options.alg_gd_step_size * gradient

                elif self.options.algorithm == "seq_nesterov":
                    for frame_index in range(self.n_frames):
                        objfcrop = self.objf[self.cropystart[frame_index][0]:self.cropyend[frame_index][0],
                                             self.cropxstart[frame_index][0]:self.cropxend[frame_index][0]]

                        # Self-calibration inner loop
                        if self.options.led_auto_calib_enabled and self.current_itr in self.options.led_auto_calib_itr_range:
                            self.findLedNaError(leds_to_process=frame_index)
                            if self.options.led_auto_calib_rigid_trans_every_led:
                                self.fitLedNaToRigidTransform(leds_to_process=range(frame_index + 1), boards_to_process=self.source_list_board_idx[frame_index])

                        if self.current_itr == 0:
                            objfcrop_hist[frame_index:, :] = objfcrop
                            gradient = self.Afunc(self.objf, idx=frame_index)[0]
                            gradient = np.reshape(gradient, [self.m_crop, self.n_crop])
                            self.objf[self.cropystart[frame_index][0]:self.cropyend[frame_index][0],
                                      self.cropxstart[frame_index][0]:self.cropxend[frame_index][0]] -= self.options.alg_nest_alpha * gradient
                        else:
                            objfcrop_d = objfcrop - objfcrop_hist[frame_index, :, :]
                            objfcrop_hist[frame_index, :, :] = objfcrop
                            objf_mom = self.objf.copy()
                            objf_mom[self.cropystart[frame_index][0]:self.cropyend[frame_index][0],
                                     self.cropxstart[frame_index][0]:self.cropxend[frame_index][0]] = objfcrop + self.options.alg_nest_beta * objfcrop_d
                            gradient = self.Afunc(objf_mom, idx=frame_index)[0].reshape([self.m_crop, self.n_crop])
                            self.objf[self.cropystart[frame_index][0]:self.cropyend[frame_index][0],
                                      self.cropxstart[frame_index][0]:self.cropxend[frame_index][0]] -= (self.options.alg_nest_alpha * gradient - self.options.alg_nest_beta * objfcrop_d)

                        if self.options.led_auto_calib_enabled:
                            if self.current_itr in self.options.led_auto_calib_itr_range:
                                self.findLedNaError(leds_to_process=frame_index)

                elif self.options.algorithm == "seq_lma_approx":

                    for frame_index in range(self.n_frames):

                        # Self-calibration inner loop
                        if self.options.led_auto_calib_enabled and self.current_itr in self.options.led_auto_calib_itr_range:
                            self.findLedNaError(leds_to_process=frame_index)
                            if self.options.led_auto_calib_rigid_trans_every_led:
                                self.fitLedNaToRigidTransform(leds_to_process=range(frame_index + 1), boards_to_process=self.source_list_board_idx[frame_index])

                        gradient = self.Afunc(self.objf, idx=frame_index)[0]
                        gradient = np.reshape(gradient, [self.m_crop, self.n_crop])
                        step_size = np.abs(self.pupil) / np.max(np.abs(self.pupil.ravel()))

                        hinv_approx = 1. / (np.abs(self.pupil) ** 2 + self.options.lm_delta2)

                        self.objf[self.cropystart[frame_index][0]:self.cropyend[frame_index][0],
                                  self.cropxstart[frame_index][0]:self.cropxend[frame_index][0]] -= step_size * hinv_approx * gradient

                elif self.options.algorithm == "seq_lma":
                    for frame_index in range(self.n_frames):

                        # Self-calibration inner loop
                        if self.options.led_auto_calib_enabled and self.current_itr in self.options.led_auto_calib_itr_range:
                            self.findLedNaError(leds_to_process=frame_index)
                            if self.options.led_auto_calib_rigid_trans_every_led:
                                self.fitLedNaToRigidTransform(leds_to_process=range(frame_index + 1), boards_to_process=self.source_list_board_idx[frame_index])

                        gradient = self.Afunc(self.objf, idx=frame_index)[0]
                        step_size = np.abs(self.pupil) / np.max(np.abs(np.reshape(self.pupil, -1)))

                        curr_hessian = lambda x: self.applyHessianSeq(x, frame_index, self.objf)
                        descent_dir = algorithms.cg(curr_hessian, -gradient, maxIter=50)[0]
                        descent_dir = np.reshape(descent_dir, [self.m_crop, self.n_crop])
                        self.objf[self.cropystart[frame_index][0]:self.cropyend[frame_index][0],
                                  self.cropxstart[frame_index][0]:self.cropxend[frame_index][0]] += 0.5 * step_size * descent_dir

                elif self.options.algorithm == "global_gd":

                    if self.current_itr > 0:  # Running this during the first iteration doesn't make sense
                        # LED Auto-calibration
                        if self.options.led_auto_calib_enabled:
                            if self.current_itr in self.options.led_auto_calib_itr_range:
                                source_list_na_prev = np.asarray(self.crop2na())
                                self.findLedNaError(leds_to_process=-1)
                                if self.options.led_auto_calib_rigid_trans:
                                    self.fitLedNaToRigidTransform()
                                self.source_list_na = np.asarray(self.crop2na())
                                print("Auto-calibration norm: %.02f" % np.linalg.norm(self.source_list_na - source_list_na_prev))

                                if self.options.led_auto_calib_add_error_na > 0.0:
                                    print("Auto-calibration error is: %.02f" % np.linalg.norm(self.source_list_na - self.source_list_na_design))

                    (grad_list, mask), fval = self.Afunc(frames_to_process=range(self.n_frames))

                    for gradient_index, gradient in enumerate(grad_list):
                        self.objf[mask[gradient_index, :, :]] += self.options.alg_gd_step_size * gradient

                elif self.options.algorithm == "global_nesterov":
                    objf_d = self.objf - objf_prev
                    objf_prev = self.objf.copy()
                    gradient = self.Afunc((self.objf + self.options.alg_nest_beta * objf_d))[0]
                    gradient = np.reshape(gradient, [self.M, self.N])
                    self.objf -= (self.options.alg_nest_alpha * gradient - self.options.alg_nest_beta * objf_d)
                    self.obj[self.current_itr + 1, :, :] = self.iF(self.objf) * (self.scale ** 2)
                    self.cost[self.current_itr + 1] = self.Afunc(self.objf, funcVal_only=True)

                    # LED Auto-calibration
                    if self.options.led_auto_calib_enabled:
                        if self.current_itr in self.options.led_auto_calib_itr_range:
                            print("Performing Auto-calibration")
                            source_list_na_prev = self.crop2na()
                            self.findLedNaError(leds_to_process=-1)
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
                    #             self.findLedNaError(leds_to_process=-1)
                    #             print("Auto-calibration norm: %.02f" % np.linalg.norm(self.source_list_na_up - self.source_list_na_prev))
                    #
                    #     self.frame_list_it += 1
                    #     self.cost[self.frame_list_it] = self.Afunc(x, funcVal_only=True)
                    #     self.objf = np.reshape(x, [self.M, self.N]).copy()
                    #     self.obj[self.frame_list_it, :, :] = self.iF(self.objf) * (self.scale ** 2)
                    #     print("|  %02d   |  %.02e  |        %.02f        |" % (self.frame_list_it, self.cost[self.frame_list_it], time.time() - t_start))
                    # x0 = self.objf.ravel()
                    # x, f, d = algorithms.lbfgs(self.Afunc, x0, iprint=1, maxiter=self.maxit-1, disp=1, callback=compute_l_bfgs_cost)
                    # iter = np.minimum(d["nit"], self.maxit)
                    # self.cost = self.cost[0:iter + 1]
                    # self.obj = self.obj[:iter + 1, :, :]

                elif self.options.algorithm == "global_newton":
                    # LED Auto-calibration
                    if self.options.led_auto_calib_enabled:
                        if self.current_itr in self.options.led_auto_calib_itr_range:
                            print("Performing Auto-calibration")
                            source_list_na_prev = self.crop2na()
                            if self.options.led_auto_calib_rigid_trans:
                                self.findLedNaError(leds_to_process=-1)
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

                # Go back to real space of the object
                self.obj = self.iF(self.objf) * (self.scale ** 2)

                # Incriment iteration count
                self.current_itr += 1

                # Compute and print cost
                self.cost[self.current_itr] = self.Afunc(funcVal_only=True, frames_to_process=range(self.n_frames))
                if not self.options.quiet:
                    if not self.options.live_plot:
                        print("|% 5d   |    %.02e    |     % 7.2f        |     % 4.2f       |" % (self.current_itr, self.cost[self.current_itr], time.time() - t_start, source_grad))
                    else:
                        self.plotter.update(self, self.current_itr)
        return


class FpmMultiSolver():
    '''
    This class is used to solve for multiple objects simultaneously using the same options structure
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

    def plotNaResult(self, index, range_to_plot=-1):
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


def rotMatrix(theta):
    M = np.array([[np.cos(theta * np.pi / 180), -np.sin(theta * np.pi / 180),0], [np.sin(theta * np.pi / 180), np.cos(theta * np.pi / 180),0],[0,0,1]])
    return(M)

def scaleMatrix(s):
    M = np.array([[1, 0, 0], [0,1,0],[0,0,1. / s]])
    return(M)

def shiftMatrix(shift):
    M = np.array([[1, 0, shift[0]], [0, 1, shift[1]], [0, 0, 1]])
    return(M)

def applyNaTransformation(source_list_na, rotation_deg=0., scale=1., shift_na=(0.,0.), flip_xy=False, flip_x=False, flip_y=False):

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

    na = source_list_na.copy()
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

    return na_2


#
# class FPMMultiplex(FpmBase):
#     """
#     Class for Multiplexed coded illumination FPM algorithms
#     """
#     def __init__(self, dataset, options):
#         FpmBase.__init__(self, dataset, options)
#         self.LEDMat_bright = data.get("LEDMat_bright")
#         self.LEDMat_dark = data.get("LEDMat_dark")
#
#     def Afunc(self, x, forward_only=False, funcVal_only=False):
#         """
#         This function computes the sequential amplitude-based cost function and gradient of FPM with respect to idx
#
#         Input:
#             x: vectorized 1D complex array of Fourier spectrum of 2D object (length self.N*self.M)
#         Returns:
#             Ax: 1D real amplitude image (shape [self.n_crop, self.m_crop])
#             fval: scalar cost function value
#             gradient: vectorized 1D complex array of gradient at f(x), Ax_idx
#         Optional parameters:
#             funcVal_only: boolean variable, function returns fval only if true
#             forward_only: boolean variable, the function returns Ax only if true, else
#                           returns both Ax and gradient
#         """
#         x = np.reshape(x, [self.N, self.M])
#         Ax = np.zeros([self.n_crop, self.m_crop, self.n_frames])
#         # Register all LED intensity images
#         objcrop_all = {}
#         for frame_index in range(self.n_frames):
#             for led_index in range(self.numlits[frame_index]):
#                 cur_key = self.cropystart[frame_index][led_index] * self.N + self.cropxstart[frame_index][led_index]
#                 if cur_key not in objcrop_all:
#                     objcrop_all[cur_key] = self.iF(x[self.cropystart[frame_index][led_index]:self.cropyend[frame_index][led_index],
#                                                      self.cropxstart[frame_index][led_index]:self.cropxend[frame_index][led_index]] * self.pupil)
#         # Apply forward model
#         for frame_index in range(self.n_frames):
#             for led_index in range(self.numlits[frame_index]):
#                 cur_key = self.cropystart[frame_index][led_index] * self.N + self.cropxstart[frame_index][led_index]
#                 Ax[:, :, frame_index] += np.abs(objcrop_all.get(cur_key)) ** 2
#         Ax = np.sqrt(Ax)
#         # print la.norm(Ax ** 2)
#         if forward_only:
#             return Ax.ravel()
#         # Compute cost
#         fval = la.norm((np.sqrt(self.frame_list) - Ax).ravel()) ** 2
#
#         # Compute gradient
#         if funcVal_only:
#             return fval
#         else:
#             gradient = np.zeros([self.N, self.M], dtype="complex128")
#             for frame_index in range(self.n_frames):
#                 numlit = self.numlits[frame_index]
#                 for led_index in range(numlit):
#                     cur_key = self.cropystart[frame_index][led_index] * self.N + self.cropxstart[frame_index][led_index]
#                     objcrop = objcrop_all[cur_key]
#                     gradient[self.cropystart[frame_index][led_index]:self.cropyend[frame_index][led_index],
#                              self.cropxstart[frame_index][led_index]:self.cropxend[frame_index][led_index]] -= \
#                         np.conj(self.pupil) * (self.F(objcrop / (Ax[:, :, frame_index] + 1e-30)
#                                                       * np.sqrt(self.frame_list[frame_index, :, :])) -
#                                                x[self.cropystart[frame_index][led_index]:self.cropyend[frame_index][led_index],
#                                                  self.cropxstart[frame_index][led_index]:self.cropxend[frame_index][led_index]] * self.pupil)
#             gradient = gradient.ravel() / float(self.n_frames)
#             return gradient, fval
#
#     def applyHessian(self, x, idx, obj):
#         """
#         This function applies the local Hessian operator to point x
#         The Hessian operator is evaluated at current point obj
#         Input:
#             x: 1D complex array point to which local Hessian is applied
#             obj: global object to be cropped corresponding to index idx
#         Return:
#             Hx: 1D complex array, same type as x
#         NOT IMPLEMENTED YET!!!!
#         """
#         pass
#
#     def run(self):
#         """
#         This function reconstructs object
#         """
#         # Compute and print cost
#         self.cost[0] = self.Afunc(self.objf, funcVal_only=True)
#         print("| Iter  |   cost     | Elapsed time (sec) |")
#         print("|  %2d   |  %.2e  |        %.2f        |" % (0, self.cost[0], 0.))
#
#         t_start = time.time()
#
#         if self.options.algorithm == "global_gd":
#             for i_it in range(self.maxit):
#                 gradient = self.Afunc(self.objf)[0]
#                 x0 = self.objf.ravel()
#                 # Tempppppp
#                 x = x0 - 10. * gradient
#                 self.objf = np.reshape(x, [self.N, self.M]).copy()
#                 self.obj[:, :, i_it+1] = self.iF(self.objf) * (self.scale ** 2)
#                 self.cost[i_it + 1] = self.Afunc(self.objf, funcVal_only=True)
#                 print("|  %2d   |  %.2e  |        %.2f        |" % (i_it + 1, self.cost[i_it + 1], time.time() - t_start))
#         elif self.options.algorithm == "global_nesterov":
#             # TEMPPPP parameter cannot be hardcoded
#             alpha = 0.5
#             beta = 0.8
#             objf_prev = self.objf
#             for i_it in range(self.maxit):
#                 objf_d = self.objf - objf_prev
#                 objf_prev = self.objf.copy()
#                 gradient = self.Afunc((self.objf+beta*objf_d))[0]
#                 gradient = np.reshape(gradient, [self.N, self.M])
#                 self.objf -= (alpha*gradient - beta*objf_d)
#                 self.obj[:, :, i_it+1] = self.iF(self.objf) * (self.scale ** 2)
#                 self.cost[i_it + 1] = self.Afunc(self.objf, funcVal_only=True)
#                 print("|  %2d   |  %.2e  |        %.2f        |" % (i_it + 1, self.cost[i_it + 1], time.time() - t_start))
#
#         elif self.options.algorithm == "global_lbfgs":
#             self.frame_list_it = 0
#
#             def compute_l_bfgs_cost(x):
#                 self.frame_list_it += 1
#                 self.cost[self.frame_list_it] = self.Afunc(x, funcVal_only=True)
#                 self.objf = np.reshape(x, [self.N, self.M]).copy()
#                 self.obj[:, :, self.frame_list_it] = self.iF(self.objf) * (self.scale ** 2)
#                 print("|  %2d   |  %.2e  |        %.2f        |" % (self.frame_list_it, self.cost[self.frame_list_it], time.time() - t_start))
#             x0 = self.objf.ravel()
#             x, f, d = algorithms.lbfgs(self.Afunc, x0, iprint=1, maxiter=self.maxit-1, disp=1, callback=compute_l_bfgs_cost)
#             print(d["task"])
#             iter = np.minimum(d["nit"], self.maxit)
#             self.cost = self.cost[0:iter+1]
#             self.obj = self.obj[:, :, 0:iter+1]
