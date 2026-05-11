'''
pipelinecone_specific_osl
---------------
Pipeline for simulating ERICA images with outer segment induced cone reflectance change
'''

import numpy as np
import os, sys, yaml
from scipy.io import loadmat
from astropy.io import fits
import matplotlib.pyplot as plt

from ERICA import generate_retina, frame_generation, AOSLO_system, sampling, peak_detection, fixation_simulation

class retina:
    def __init__(self, current_directory: str):

        # config relavent
        self.current_directory = current_directory
        self.cone_reflectance_parameters : dict = None
        self.myAOSLO : object = None
        self.field_of_view = None
        self.eccentricity = None

        # retina grid
        self.cone_positions_window = None  # in um       
        self.cone_types = None
        self.osl_cone_map = None 

        # fovea
        self.cone_positions_foveated = None

        
    def load_config(self, isetbio=True):
        '''
        Load AOSLO system parameters and cone reflectance parameters from .yaml config file

        Parameters
        ----------
        - isetbio: bool
            Whether or not load isetbio mosaic and overwrite system parameters.

        Reads
        ------
        - self.current_directory

        Modifies
        ---------
        - self.cone_refelctance_parameters
        - self.myAOSLO
        - self.eccentricity
        - self.field_of_view
        - self.mosaic
        '''

        # AOSLO system parameters
        # system_parameters_file = 'system_parameters_demo.yaml'
        system_parameters_file = 'system_parameters_myAOSLO.yaml'
        system_parameters_file = os.path.normpath(os.path.join(self.current_directory, 'System_parameters', system_parameters_file))
        sys.stdout.write('System parameters file: %s\n'%system_parameters_file)
        self.myAOSLO = AOSLO_system.AOSLO(system_parameters_file)

        # Cone reflectance parameters
        cone_reflectance_parameters_file = 'cone_reflectance_parameters.yaml'
        cone_reflectance_parameters_file = os.path.normpath(os.path.join(self.current_directory, 'Reflectance_parameters', cone_reflectance_parameters_file))
        sys.stdout.write('Cone reflectance parameter file: %s\n'%cone_reflectance_parameters_file)
        # Open the cone reflectance parameters file
        with open(cone_reflectance_parameters_file, 'r') as f:
            self.cone_reflectance_parameters = yaml.safe_load(f)

        # fixational eye movements parameters
        fixation_parameters_file = 'fixation_parameters_default.yaml'
        fixation_parameters_file = os.path.normpath(os.path.join(self.current_directory, 'Fixation_parameters', 'fixation_parameters_default.yaml'))
        sys.stdout.write('Eye movements parameter file: %s\n'%fixation_parameters_file)
        self.myEyeMotion = fixation_simulation.Fixation(fixation_parameters_file, self.myAOSLO.time_resolution)

        if isetbio == True:
            # load grid
            mosaic_number = self.myAOSLO.parameters['mosaic_number']
            mosaic_folder = self.myAOSLO.parameters['mosaic_folder']
            mosaic_file = os.path.normpath(os.path.join(mosaic_folder, f'mosaic_{mosaic_number:02d}.mat'))
            mosaic = loadmat(mosaic_file, squeeze_me=True, struct_as_record=False)
            self.mosaic = mosaic['mosaic']
            print('Unscaled mosaic grid file: %s'%mosaic_file)

            # overwrite params
            self.myAOSLO.parameters['microns_per_degree'] = self.mosaic.microns_per_deg 
            self.myAOSLO.parameters['eccentricity_y_x'] = self.mosaic.ecc_deg_y_x

        self.eccentricity = self.myAOSLO.parameters["eccentricity_y_x"]
        self.field_of_view = self.myAOSLO.parameters["field_of_view_y_x"]
        

    def compute_mosaic_grid(self, eccentricity=None, field_of_view=None):
        # Here the eccentricity and fov are for generating the groundtruth mosaic grid. For generating ERICA image (not the image from ``sampling.quickLookImaging()``) which has scanning characteristic, the field of view is limited by the AOSLO sampling rate, have nothing do with the eccentricity and fov input here. 
        # Mosaic generated from this function seems different from the ``sample_from_foveate_mosaic()``. 
        # The mosaic grid here is better as large as possible so that eye movement can be capture. 
        '''
        Compute mosaic grid from given eccentricity and field of view
        This use ERICA reaction diffusion output. 
        
        Reads
        ------
        - self.myAOSLO.parameters
        
        Modifies
        --------
        - self.eccentricity
        - self.field_of_view
        - self.cone_positions_window
        - self.cone_sizes_window
        '''

        if eccentricity is None:
            print('Using eccentricity from config.')
            eccentricity = self.eccentricity

        if field_of_view is None:
            print('Using field of view from config.')
            field_of_view = self.field_of_view

        # Load pre-generated grid (reaction diffusion output)
        mosaic_number = self.myAOSLO.parameters['mosaic_number']
        peaks = fits.getdata(os.path.normpath(os.path.join(self.current_directory, 'Reaction_diffusion_outputs/mosaic_%03i.fits'%mosaic_number)))

        # Scale the points to a fixed spacing, determined by the eccentricity
        rescaled_cone_positions = generate_retina.scaleRetina(np.transpose(peaks), eccentricity, self.myAOSLO.parameters['microns_per_degree'])

        self.cone_positions_window = rescaled_cone_positions[:,np.where(
            (rescaled_cone_positions[0] > (eccentricity[0] - 0.5*field_of_view[0]) * self.myAOSLO.parameters['microns_per_degree']) 
            & (rescaled_cone_positions[0] < (eccentricity[0] + 0.5*field_of_view[0]) * self.myAOSLO.parameters['microns_per_degree']) 
            & (rescaled_cone_positions[1] > (eccentricity[1] - 0.5*field_of_view[1]) * self.myAOSLO.parameters['microns_per_degree']) 
            & (rescaled_cone_positions[1] < (eccentricity[1] + 0.5*field_of_view[1]) * self.myAOSLO.parameters['microns_per_degree']))[0]]
        
        # Check the spacing of the generated cones
        distances, indices = peak_detection.findNeighbours(np.transpose(rescaled_cone_positions), number_neighbours=6)
        #  Take the median distance between a cone and its 3 nearest neighbours
        spacing_points = np.median(distances[:,1:4], axis=1)

        # Check the required spacing based on histology
        spacing_histology = generate_retina.spacingFromEccentricity(eccentricity)

        np.testing.assert_almost_equal(spacing_points.mean(),spacing_histology,decimal=3,
                                    err_msg=f"Simulated cone spacing ({spacing_points.mean():.03f}) does not match with histological values ({spacing_histology:.03f})")
        
        # Calculate the eccentricities of the cells
        eccentricities = np.sqrt(self.cone_positions_window[0]**2 + self.cone_positions_window[1]**2) / self.myAOSLO.parameters['microns_per_degree']

        # Calculate the cone sizes
        cone_sizes_window = generate_retina.coneDiameter(eccentricities, self.myAOSLO.parameters['microns_per_degree'], self.myAOSLO.parameters['wavelength_m'] )
        print(cone_sizes_window.shape)

        # To make cells that are circular, copy the cone size for both the horizonal and vertical dimensions. The
        # following creates a (2, N) array of the vertical and horizontal widths
        self.cone_sizes_window = np.repeat(cone_sizes_window[np.newaxis,:], 2, axis=0)

    def compute_cone_specific_properties(self, cone_positions_um=None):
        '''
        Assign cone type to mosaic grid based on eccentricity, and
        generate outer segment length based on cone type.
        '''
        from .utils import assign_cone_types, generate_cone_types_osl

        if cone_positions_um is None:
            cone_positions_um = self.cone_positions_window
        else:
            cone_positions_um = cone_positions_um

        self.cone_types = assign_cone_types(cone_positions_um)
        self.osl_cone_map = generate_cone_types_osl(self.cone_types)

    def init_grid(self):
        self.load_config()
        self.compute_mosaic_grid()
        self.compute_cone_specific_properties()

    def _ensure_grid_init(self, auto_init: bool = True):
        import warnings
        '''
        Ensure that grid-related states are initialized.
        Raises or initializes depending on `auto_init`.
        '''
        missing = []
        if self.cone_reflectance_parameters is None:
            missing.append("cone_reflectance_parameters")
        if self.myAOSLO is None:
            missing.append("myAOSLO")
        if self.cone_positions_window is None:
            missing.append("cone_positions_window")
        if self.osl_cone_map is None:
            missing.append("osl_cone_map")

        if missing:
            msg = f"Mosaic grid not initialized\n(missing: {', '.join(missing)})."
            if auto_init:
                warnings.warn(msg + "\nAuto-running init_grid().", RuntimeWarning, stacklevel=2)
                self.init_grid()
            else:
                raise RuntimeError(msg + " Call init_grid() first.")
        else:
            print("Grid has been properly initialised.")

    def generate_mosaic(self, cone_positions=None, cone_sizes=None, input_map=None, eccentricity=None, field_of_view=None, auto_init:bool=True, save_retina:bool=True):

        '''
        Compute cone-specific properties and make grid to ground-truth retinal mosaic image
        with no scanning characteristics, diffraction and aberration, electronic noise etc.

        Parameters
        ----------
        - cone_positions : ndarray
            Cone position in microns for the targeted field of view and eccentricity.
            Default is None, using ``self.cone_positions_window``
        - cone_sizes : ndarray
            Cone sizes window in microns for the targeted field of view and eccentricity.
            Default is None, using ``self.cone_sizes_window``
        - input_map : ndarray
            Outer segment length map to use. 
            If None, use self.osl_cone_map.
        - eccentricity :
            Eccentricity that match the given cone position windows.
            Default is None, using ``self.eccentriicty`` (from config).
        - field_of_view :
            Field of view that match the given cone position windows. 
            Default is None, using ``self.field_of_view`` (from config).
        - auto_init : bool
            Whether to auto-run init_grid() if grid/map is missing.
        - save_retina : bool
            Whether to write retina_window to ``self.retina_window``
            Default is True. 
        
        Reads
        ------
        - self.cone_reflectance_parameters
        - self.myAOSLO.parameters

        Returns
        -------
        - image : ndarray
            Groundtruth retina image without diffraction limited
        - retina_window : ndarray
            Parameters output from ``generate_retina.retina()``.
            Return only when ``save_retina`` is False.
        '''

        if cone_positions is None:
            self._ensure_grid_init(auto_init=auto_init)
            print('Using cone position window from object.')
            cone_positions = self.cone_positions_window
        
        if cone_sizes is None:
            print('Using cone size window from object.')
            cone_sizes = self.cone_sizes_window

        if input_map is None:
            print('Using OSL map from object.')
            input_map = self.osl_cone_map
        
        if eccentricity is None:
            print('Using eccentricity from config.')
            eccentricity = self.eccentricity

        if field_of_view is None:
            print('Using field of view from config.')
            field_of_view = self.field_of_view
        

        retina_window = generate_retina.retina(cone_locations_um=cone_positions,
                                            cone_sizes_um=cone_sizes,
                                            outer_segment_length_um=input_map,
                                            width_variation=self.cone_reflectance_parameters['width_variation'],
                                            distribution=self.cone_reflectance_parameters['distribution'],
                                            wavelength_nm=self.myAOSLO.parameters['wavelength_m']*1e9)

        image, groundtruth_array = sampling.quickLookImage(retina_window, 
                                                            eccentricity, 
                                                            field_of_view, 
                                                            self.myAOSLO.parameters['microns_per_degree'], 
                                                            self.myAOSLO.parameters['pixel_size_arcmin_y_x'])
        
        if save_retina == True:
            print('``retina_window`` has been write to ``self.retina_window``.')
            self.retina_window = retina_window
            return image
        elif save_retina == False:
            return image, retina_window

        
    def diffraction_limited(self, image):
        '''
        Wrapper for adding diffraction to the image.
        '''
        image_diffraction = frame_generation.diffractionOnly(image, 
                                                   pixel_size_arcmin=self.myAOSLO.parameters['pixel_size_arcmin_y_x'],
                                                   wavelength=self.myAOSLO.parameters['wavelength_m'],
                                                   pupil_diameter=self.myAOSLO.parameters['pupil_diameter_m'],
                                                   pinhole_add=self.myAOSLO.parameters['pinhole_diameter_Airy_radii'])
        return image_diffraction
    
    def add_noise(self, image):
        '''
        Using skimage to add noise to the image.
        '''
        import skimage
        image_noisy = skimage.util.random_noise(image, 
                                            mode=self.myAOSLO.parameters['noise_mode'], clip=True, 
                                            mean=self.myAOSLO.parameters['noise_mean_normalised_graylevels'], 
                                            var=self.myAOSLO.parameters['noise_stdev_normalised_graylevels']**2)
        return image_noisy
    
    def two_session_imaging(self, start_time:str, interval_h:float, auto_init:bool = True, visualised:bool = True):

        '''
        Demo of images at two different acquisition times. 
        Diffraction-limited images and gaussian noise are added and compared.
        Images do not have scanning characteristics. 
        
        Parameters
        ----------
        - start_time : str
            HH:MM string of start time
        - interval_h : float
            Interval in hour of the second session.
        
        Reads
        -----
        - self.osl_cone_map
        - self.cone_position_windows
        - self.cone_sizes_window_2d
        '''

        # self._ensure_grid_init(auto_init=auto_init)

        # session 1
        image1 = self.generate_mosaic(auto_init=auto_init, save_retina=False)
        image1_diffraction = self.diffraction_limited(image1)
        image1_noisy = self.add_noise(image1_diffraction)
        session_1 = {
            "mosaic": image1,
            "diffraction_limit": image1_diffraction,
            "noisy": image1_noisy,
            "osl_map": self.osl_cone_map,
        }

        # session 2
        from .utils import osl_natural
        osl_renewed = osl_natural(self.osl_cone_map, start_time, interval_h)
        image2 = self.generate_mosaic(input_map=osl_renewed, save_retina=False)
        image2_diffraction = self.diffraction_limited(image2)
        image2_noisy = self.add_noise(image2_diffraction)
        session_2 = {
            "mosaic": image2,
            "diffraction_limit": image2_diffraction,
            "noisy": image2_noisy,
            "osl_map": osl_renewed,
        }

        bundle = {
            "session1": session_1,
            "session2": session_2,
            "start_time": start_time,
            "interval_h": interval_h,
        }

        if visualised == True:
            from .visualisation import visualise_two_session_imaging
            visualise_two_session_imaging(image1_diffraction, image2_diffraction,
                                          image1_noisy, image2_noisy, 
                                          interval_h, self.myAOSLO.parameters['noise_mode'],
                                          self.osl_cone_map, osl_renewed)
            
        return bundle

    def load_foveate_mosaic_grid(self, eccentricity=None):

        '''
        Load pre-generated mosaic and rescale to fovea. 
        Similar to compute_mosaic_grid(), without eccentricity and field of view sampling.
        Output a near 12 x 12 degree mosaic grid. 
        
        Reads
        -----
        - self.myAOSLO.parameters

        Modifies
        --------
        - self.cone_positions_foveated
        - self.cone_sizes_foveated
        '''

        if eccentricity is None:
            eccentricity = self.eccentricity

        if self.cone_positions_foveated is not None:
            print('Foveated grid has already been loaded.')
            return
        else: 

            foveated_points = generate_retina.foveatePoints(self.peaks, self.myAOSLO.parameters['microns_per_degree'], eccentricity=eccentricity)
            self.cone_positions_foveated = generate_retina.centreFovea(foveated_points)

            # Calculate the eccentricities of the cones
            eccentricities_foveated = np.sqrt(self.cone_positions_foveated[0]**2 + self.cone_positions_foveated[1]**2) / self.myAOSLO.parameters['microns_per_degree']

            # Calculate the cone sizes (standard deviation of Gaussian intensity profile)
            cone_size_foveated = generate_retina.coneDiameter(eccentricities_foveated, self.myAOSLO.parameters['microns_per_degree'], self.myAOSLO.parameters['wavelength_m'])

            self.cone_sizes_foveated = np.repeat(cone_size_foveated[np.newaxis,:], 2, axis=0)

    def sample_from_foveate_mosaic(self, eccentricity=None, field_of_view=None):

        # TODO: 
        # - add eccentricity bound beyond which is out of range.
        # - deal with repeat process with ``compute_cone_specific_properties()``
        '''
        Sample a field of view at given eccentricity from the foveate mosaic. 
        
        Reads
        ------
        - self.myAOSLO.parameters
        - self.cone_positions_foveated

        Modifies
        - self.cone_positions_foveated_window
        - self.cone_sizes_foveated_window
        '''

        if eccentricity == None:
            # Use eccentricity from config
            eccentricity = self.eccentricity

        if field_of_view == None:
            # Use field of fiew from config
            field_of_view = self.field_of_view
        
        cone_indices_window = np.where(
        (self.cone_positions_foveated[0] > (eccentricity[0] - 0.5 * field_of_view[0]) * self.myAOSLO.parameters['microns_per_degree']) 
        & (self.cone_positions_foveated[0] < (eccentricity[0] + 0.5 * field_of_view[0]) * self.myAOSLO.parameters['microns_per_degree']) 
        & (self.cone_positions_foveated[1] > (eccentricity[1] - 0.5 * field_of_view[1]) * self.myAOSLO.parameters['microns_per_degree']) 
        & (self.cone_positions_foveated[1] < (eccentricity[1] + 0.5 * field_of_view[1]) * self.myAOSLO.parameters['microns_per_degree']))[0]

        self.cone_positions_foveated_window = self.cone_positions_foveated[:,cone_indices_window]
        self.cone_sizes_foveated_window = self.cone_sizes_foveated[:, cone_indices_window]

        from .utils import assign_cone_types, generate_cone_types_osl
        self.cone_types_foveated = assign_cone_types(self.cone_positions_foveated_window)
        self.osl_cone_map_foveated = generate_cone_types_osl(self.cone_types_foveated)

    

    def make_aoslo_image(self, retina_params=None, eye_motion_arcmin=None, eccentricity=None, noise:bool=True, clip:bool=False):

        '''
        Generate AOSLO images.
        This does not required FOV because the image size is determined by 
        the AOSLO sampling rate. 

        Parameters
        -----------
        - retina_params : ndarray
            The output from generate_retina.retina(). Array dimensions is [6, n_cones].
            Default is None, using ``self.retina_window``.
        - eye_motion_arcmin : ndarray
            Eye motion trace in arcmin. 
            Default is None, using static eye.
        - eccentricity : ndarray
            The eccentricity that matched the given retina_params. 
            Default is None, using ``self.eccentricity`` (from the config)
        - noise : bool
            Whether or not adding Gaussian white noise.
            Default is True.
        - clip : bool
            Whether or not clip the intensity by the saturation value
            (from ``self.myAOSLO.parameters['proportion_saturated_pixels']``).
            Default is False.
        
        Returns
        --------
        - image : ndarray
            Image that is desinusoidded and flyback cropped, 
            with diffraction limited (i.e. convolved with system PSF).

        Reads
        -----
        - self.myAOSLO
        '''
        
        if retina_params is None:
            retina_params = self.retina_window
        
        if eye_motion_arcmin is None:
            eye_motion_arcmin = np.zeros((2, self.myAOSLO.slow.shape[0]))

        if eccentricity is None:
            eccentricity = self.eccentricity
        
        intensity, *_ = sampling.sampleRetina(retina_params, eye_motion_arcmin, self.myAOSLO.fast, self.myAOSLO.slow, self.myAOSLO.parameters['pixel_size_arcmin_y_x'], self.myAOSLO.parameters['microns_per_degree'], eccentricity)

        fr_image = intensity.reshape((self.myAOSLO.parameters['number_scan_lines_including_flyback'],self.myAOSLO.parameters['number_pixels_per_line']*2))

        image = fr_image[:,:self.myAOSLO.parameters['number_pixels_per_line']]

        # add shot noise before desinusoid
        if noise == True:
            eps = 1e-18
            image_norm =  (image-image.min())/(image.max()-image.min()+eps)
            # image_norm =  (image)/(image.max()+eps)
            flat = image.flatten()
            noise_sample = np.random.normal(self.myAOSLO.parameters['noise_mean_normalised_graylevels'], self.myAOSLO.parameters['noise_stdev_normalised_graylevels'], size=flat.shape[0])
            noise_sample = noise_sample.reshape(image.shape)

            image = image_norm + noise_sample 

        image_desinusoid = self.myAOSLO.desinusoid(image)

        nlines_flyback = self.myAOSLO.parameters['number_scan_lines_including_flyback'] - self.myAOSLO.parameters['number_scan_lines_excluding_flyback']

        image_desinusoid = image_desinusoid[:-nlines_flyback]

        image_dffr = self.diffraction_limited(image_desinusoid)

        if clip == True:
            saturation_val = np.percentile(image_dffr, (1.0-self.myAOSLO.parameters['proportion_saturated_pixels'])*100)
            image_dffr = np.clip(image_dffr, 0.0, saturation_val)

        return image_dffr
    

    def drift_trace_one_frame(self):
        '''
        Copied from villmaria/ISETBioImageGeneration/ERICAMethods/RandomAnglesDrift_ISETBioMosaic_ForwardBackward.py

        Generates a drift motion trace for constructing a single aoslo frame. 
        
        Reads
        -----
        - self.myAOSLO
        - self.myEyeMotion    
            
        Returns
        - microsaccade : ndarray
            An array of eye motion in arcmin.
        
        '''
        from ORG.utils import generate_drift
        # Pick a random angle in degrees, convert it to radians 
        microsaccade_angle = np.random.uniform(0, 360)
        microsaccade_angle *= (np.pi/180)
        # Pick a microsaccade amplitude in ArcMin 
        microsaccade_amplitude = np.random.uniform(0, 30)
        
        # microsaccade_amplitude = 0 
        # Make it a nice round number :-)
        microsaccade_amplitude = round(microsaccade_amplitude, 0)
        total_time = self.myAOSLO.time_resolution * self.myAOSLO.n_samples
        # The number of samples in that time period
        n_samples_microsaccade = int(np.ceil(total_time / float(self.myAOSLO.time_resolution)))
        print(f'n samples: {n_samples_microsaccade}')

        if microsaccade_amplitude > 0:
            # Log of the amplitude in degrees
            l = np.log10(microsaccade_amplitude/60.)

            # Estimate the maximum speed from the main sequence 
            ms = l + np.log10(self.myEyeMotion.parameters['Main_sequence_factor'])
            microsaccade_max_speed = 60 * 10**ms       
            
            # Generate the microsaccade
            microsaccade = fixation_simulation.microsaccade(microsaccade_amplitude, microsaccade_angle, microsaccade_max_speed, self.myAOSLO.time_resolution, n_samples=n_samples_microsaccade)

            # Calculate the velocity
            velocity =(microsaccade[:,1:] - microsaccade[:,:-1]) / self.myAOSLO.time_resolution

            radial_velocity = np.sqrt(velocity[1]**2 + velocity[0]**2)
            microsaccade_start = np.where(radial_velocity>radial_velocity.max()*0.1)[0][0]
            microsaccade_end = radial_velocity.shape[0] - np.where(radial_velocity[::-1]>radial_velocity.max()*0.1)[0][0]
            print(f'microsaccade start: {microsaccade_start}')
            print(f'microsaccade end: {microsaccade_end}')

            calc_drift = np.random.uniform(3.8, 12.0)
            total_time_drift = 2/30.
            n_samples_drift = int(np.ceil(total_time_drift / float(self.myAOSLO.time_resolution)))
            full_drift = (total_time_drift / 0.5) * calc_drift
            drift_motion = generate_drift(n_samples_drift, full_drift)
            print(f'n samples drift: {n_samples_drift}')
            print(f'drift motion size: {drift_motion.shape}')

            microsaccade_copy = np.copy(microsaccade)
            
            drift_sequence_1 = drift_motion[:,:microsaccade_start]
            drift_sequence_1[0] = drift_sequence_1[0] - drift_sequence_1[0,-1]
            drift_sequence_1[1] = drift_sequence_1[1] - drift_sequence_1[1,-1]

            drift_sequence_1[0] = drift_sequence_1[0] + microsaccade_copy[0, microsaccade_start-1]
            drift_sequence_1[1] = drift_sequence_1[1] + microsaccade_copy[1, microsaccade_start-1]

            print(f'drift sequence1 shape: {drift_sequence_1.shape}')

            microsaccade_copy[:,:microsaccade_start] = drift_sequence_1

            calc_drift2 = np.random.uniform(3.8, 12.0)
            total_time_drift = 2/30.
            n_samples_drift = int(np.ceil(total_time_drift / float(self.myAOSLO.time_resolution)))
            full_drift = (total_time_drift / 0.5) * calc_drift2
            drift_motion = generate_drift(n_samples_drift, full_drift)

            drift_sequence_2 = drift_motion[:,:len(microsaccade[0,microsaccade_end:])]
            drift_sequence_2[0] = drift_sequence_2[0] - drift_sequence_2[0,0]
            drift_sequence_2[1] = drift_sequence_2[1] - drift_sequence_2[1,0]

            drift_sequence_2[0] = drift_sequence_2[0] + microsaccade_copy[0, microsaccade_end]
            drift_sequence_2[1] = drift_sequence_2[1] + microsaccade_copy[1, microsaccade_end]
            microsaccade_copy[:,microsaccade_end:] = drift_sequence_2

            print(f'drift sequence2 shape: {drift_sequence_2.shape}')

            microsaccade = microsaccade_copy[:,:self.myAOSLO.fast.shape[0]]

            dift_seq = [calc_drift, calc_drift2]

        else: 
            calc_drift = np.random.uniform(3.8, 12.0)
            dift_seq = [calc_drift]
            n_samples_drift = int(np.ceil(total_time / float(self.myAOSLO.time_resolution)))
            full_drift = (total_time / 0.5) * calc_drift
            drift_motion = generate_drift(n_samples_drift, full_drift)
            microsaccade = drift_motion
            microsaccade_max_speed = 0.

        return microsaccade
    
    def em_trace_full(self, total_time):
        '''
        Simulate and segmentate a motion trace with the given duration
        
        :param self: Description
        :param total_time: Description
        '''

        motion_x, motion_y, time_samples, *_ = self.myEyeMotion.fixationalEyemovementTrace(total_time)

        # Split the full eye movement trace into each frame
        # Will need to drop some samples to make round frame
        n_samples = time_samples.shape[0]
        n_frames = int(n_samples / self.myAOSLO.n_samples)
        n_samples_rounded = n_frames * self.myAOSLO.n_samples

        samples = np.full(n_frames, n_samples_rounded // n_frames)
        samples[:n_samples_rounded % n_frames] += 1

        if all(x == self.myAOSLO.n_samples for x in samples) is not True:
            raise ValueError(f'Cannot make samples per frame equal to AOSLO samples per frame {self.myAOSLO.n_samples}')

        motions = []
        c = 0
        for t in samples:
            framewise_motion = np.zeros((2,samples[0]))
            framewise_motion[0] = motion_y[c:c+t]
            framewise_motion[1] = motion_x[c:c+t]
            motions.append(framewise_motion)
            c += t
        
        return motions
    