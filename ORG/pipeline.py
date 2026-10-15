'''
pipeline.py
---------------
Pipeline for simulating ERICA images with outer segment length induced cone reflectance change, using ISETBio mosaic.

Sirui Liu, Jan 2026
'''

import numpy as np
import os, sys, yaml

from ERICA import frame_generation, AOSLO_system, sampling, fixation_simulation

class retina:
    def __init__(self, current_directory: str):

        # config relavent
        self.current_directory = current_directory
        self.cone_reflectance_parameters : dict = None
        self.myAOSLO : object = None
        self.eccentricity = None

        
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
        - self.mosaic
        '''

        # AOSLO system parameters
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
            import mat73
            mosaic_number = self.myAOSLO.parameters['mosaic_number']
            mosaic_folder = self.myAOSLO.parameters['mosaic_folder']
            mosaic_file = os.path.normpath(os.path.join(mosaic_folder, f'mosaic_{mosaic_number:02d}.mat'))
            mosaic = mat73.loadmat(mosaic_file)
            self.mosaic = mosaic['mosaic']
            print('Unscaled mosaic grid file: %s'%mosaic_file)

            # overwrite params
            self.myAOSLO.parameters['microns_per_degree'] = self.mosaic['microns_per_deg'] 

        self.eccentricity = self.myAOSLO.parameters["eccentricity_y_x"]     
        return   
        
    def isetbio_mosaic(self):
        '''
        Generate retina from ISETbio mosaic. 
        Analogous to `ERICA.generate_retina.retina()`.
        Adapted from villmaria/ISETBioImageGeneration/ERICAMethods/RandomAnglesDrift_ISETBioMosaic_ForwardBackward.py
        
        Reads
        -----
        - self.mosaic: ISETbio mosaic
        - self.myAOSLO

        Modifies
        --------
        - self.retina_window_isetbio
        '''
        n_cones = self.mosaic['cone_types'].shape[0]

        # Generate rotation angles
        rotation_angles = np.random.uniform(0, 2*np.pi, size=(n_cones))

        if self.cone_reflectance_parameters['distribution'] == 'interference':
            # Two beam interference reflectance
            NO = 1.430  # Refractive index of outer segment
            opd_nm = 2 * NO * self.mosaic['osl_um'] * 1e3    # OPD = 2 * n * l
            d_phi = 2 * np.pi * opd_nm / self.myAOSLO.parameters['wavelength_m']*1e-9   # phase shift = 2pi * OPD / lambda
            reflectance = 0.5 * (1 + np.cos(d_phi))      # two beam interference, assuming I1=I2=0.5

        elif self.cone_reflectance_parameters['distribution'] == 'normal':

            mean_reflectance = self.cone_reflectance_parameters['normal_mean']
            std_reflectance = self.cone_reflectance_parameters['normal_std']

            if type(mean_reflectance) == type(None) or type(std_reflectance) == type(None):
                raise ValueError('The mean and standard deviation of the normal distribution for cone reflectance selection must be specified')
            else:
                reflectance = np.random.normal(mean_reflectance, std_reflectance, size=(n_cones))   

        retina_window = np.zeros((6, n_cones))
        # Y-coordinates
        retina_window[0] = np.atleast_1d(self.mosaic['cone_position_um'][:,0]).reshape(1, -1)
        # X-coordinates
        retina_window[1] = np.atleast_1d(self.mosaic['cone_position_um'][:,1]).reshape(1, -1)
        # cone characteristic radius in Y (sigma_y)
        retina_window[2] = np.atleast_1d(self.mosaic['cone_aperture_um'] * self.mosaic['aperture2gaussian_sigma']).reshape(1, -1)
        # cone characteristic radius in X (sigma_x)
        retina_window[3] = np.atleast_1d(self.mosaic['cone_aperture_um'] * self.mosaic['aperture2gaussian_sigma']).reshape(1, -1)
        # rotation angles / orientations (theta)
        retina_window[4] = np.atleast_1d(rotation_angles).reshape(1, -1)
        # cone reflectance
        retina_window[5] = np.atleast_1d(reflectance).reshape(1, -1)
        self.retina_window_isetbio = retina_window

        return
        
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
    
    def add_noise2image(self, image):
        '''
        Using skimage to add noise to the image.
        '''
        import skimage
        # Add Poisson Noise first (Signal-dependent)
        poisson_img = skimage.util.random_noise(image, mode='poisson')

        # Add Gaussian Noise second (Additive read noise)
        image_noisy = skimage.util.random_noise(poisson_img, 
                                            mode=self.myAOSLO.parameters['noise_mode'], clip=True, 
                                            mean=self.myAOSLO.parameters['noise_mean_normalised_graylevels'], 
                                            var=self.myAOSLO.parameters['noise_stdev_normalised_graylevels']**2)
        return image_noisy 
    
    def add_noise2stream(self, 
                         image_stream, 
                         noise_mean=None, 
                         noise_stdev=None,
                         vals=None,
                         poisson=False,
                         ):
        '''
        Add Poisson noise and Gaussian white noise to data stream (before desinusoid).
        '''
        if noise_mean is None:
            noise_mean = self.myAOSLO.parameters['noise_mean_normalised_graylevels']
        if noise_stdev is None:
            noise_stdev = self.myAOSLO.parameters['noise_stdev_normalised_graylevels']
        if vals is None:
            vals = self.myAOSLO.parameters['poisson_scale']

        if poisson is True:
            # shot noise, adapted from skimage
            noisy_stream = np.random.poisson(image_stream * vals) / float(vals)
        else:
            noisy_stream = image_stream

        noise_sample = np.random.normal(noise_mean, noise_stdev, size=(image_stream.ravel()).shape[0])  # noise floor
        noise_sample = noise_sample.reshape(image_stream.shape)

        noisy_stream = noisy_stream + noise_sample
        
        return noisy_stream

    def make_aoslo_image(self, 
                         retina_params=None, 
                         eye_motion_arcmin=None, 
                         eccentricity=None, 
                         noise:bool=True, 
                         clip:bool=False, 
                         noise_mean=None, 
                         noise_stdev=None,
                         poisson=False,
                         return_cones=None,
                         return_intensity=None,
                         ):

        '''
        Generate AOSLO images.
        This does not required FOV because the image size is determined by 
        the AOSLO sampling rate. 

        Parameters
        -----------
        - retina_params : ndarray
            The output from generate_retina.retina(). Array dimensions is [6, n_cones].
            Default is None, using `self.retina_window`.
        - eye_motion_arcmin : ndarray
            Eye motion trace in arcmin. 
            Default is None, using static eye.
        - eccentricity : ndarray
            The eccentricity that matched the given retina_params. 
            Default is None, using `self.eccentricity` (from the config)
        - noise : bool
            Whether or not adding Gaussian white noise.
            Default is True.
        - clip : bool
            Whether or not clip the intensity by the saturation value
            (from `self.myAOSLO.parameters['proportion_saturated_pixels']`).
            Default is False.
        - noise_mean : float
            Mean of Gaussian noise normalised graylevels. Defaults None, use `myAOSLO.parameters`
        - noise_stdev : float
            Standard deviation of Gaussian noise normalised graylevels. Defaults None, use `myAOSLO.parameters`
        
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
            retina_params = self.retina_window_isetbio
        
        if eye_motion_arcmin is None:
            eye_motion_arcmin = np.zeros((2, self.myAOSLO.slow.shape[0]))

        if eccentricity is None:
            eccentricity = self.eccentricity
        
        intensity, groundtruth_centroids, _, _, cones_scanned = sampling.sampleRetina(retina_params, eye_motion_arcmin, self.myAOSLO.fast, self.myAOSLO.slow, self.myAOSLO.parameters['pixel_size_arcmin_y_x'], self.myAOSLO.parameters['microns_per_degree'], eccentricity)

        # add noise before desinusoid
        if noise == True:
            intensity = self.add_noise2stream(intensity, noise_mean, noise_stdev, poisson=poisson)

        # reshape 1D intensity array to 2D image, fr_image contain forward and backward scan
        fr_image = intensity.reshape(
            (self.myAOSLO.parameters['number_scan_lines_including_flyback'],
             self.myAOSLO.parameters['number_pixels_per_line']*2)
             )  

        # crop so keep forward scan only
        image = fr_image[:,:self.myAOSLO.parameters['number_pixels_per_line']]

        image_desinusoid = self.myAOSLO.desinusoid(image)

        # crop flyback
        nlines_flyback = self.myAOSLO.parameters['number_scan_lines_including_flyback'] - self.myAOSLO.parameters['number_scan_lines_excluding_flyback']
        image_desinusoid = image_desinusoid[:-nlines_flyback]

        image_dffr = self.diffraction_limited(image_desinusoid)

        if clip == True:
            saturation_val = np.percentile(image_dffr, (1.0-self.myAOSLO.parameters['proportion_saturated_pixels'])*100)
            image_dffr = np.clip(image_dffr, 0.0, saturation_val)

        if return_intensity is not None and return_cones is None:
            return image_dffr, intensity
        
        if return_intensity is None and return_cones is not None:
            return image_dffr, cones_scanned, groundtruth_centroids
        
        if return_intensity is not None and return_cones is not None:
            return image_dffr, intensity, cones_scanned, groundtruth_centroids

        return image_dffr
    
    def make_montage(self, retina_window=None, eccentricity=None, field_of_view=None, noise:bool=True):
        '''
        Scan a bigger area as montage.

        '''
        if retina_window is None:
            retina_window = self.retina_window_isetbio
        if eccentricity is None:
            eccentricity = self.eccentricity
        if field_of_view is None:
            field_of_view = self.myAOSLO.parameters['montage_field_of_view_y_x']

        print(f'eccentricity: {eccentricity}')
        print(f'field of view: {field_of_view}')

        image, groundtruth, cones_scanned = sampling.quickLookImage(retina_window, eccentricity, field_of_view, microns_per_degree=self.myAOSLO.parameters['microns_per_degree'], pixel_size_arcmin=self.myAOSLO.parameters['pixel_size_arcmin_y_x'])

        if noise == True:
            image = self.add_noise2image(image)

        montage = self.diffraction_limited(image)

        return montage, groundtruth, cones_scanned

    
    def em_trace_full(self, total_time):
        '''
        Simulate and segmentate a motion trace with the given duration.
        ERICA built-in eye movement trace is for given time, but the number
        of samples is rounded. 
        This function is to segment a continous eye motion into eye motion
        per frame.
        
        Parameters:
            total_time (float) : eye movement trace time length (s).

        Returns:
            motions (list) : yx motion of n_frames
        '''

        motion_x, motion_y, *_ = self.myEyeMotion.fixationalEyemovementTrace(total_time)

        # Split the full eye movement trace into each frame
        # Will need to drop some samples to make round frame
        n_samples = motion_x.shape[0]
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
    
    
    def prepare_trial_params(self, 
                             condition=4, 
                             total_time=None, 
                             em_free=False, 
                             coherency=False,
                             ):

        if total_time is None:
            total_time = self.cone_reflectance_parameters['duration_s']
        
        if condition == 4:
            exp_type = 'static'
        else:
            exp_type = 'flicker'

        frame_time = self.myAOSLO.n_samples * self.myAOSLO.time_resolution
        n_frames = int(total_time / frame_time)

        if em_free is False:
            motions = self.em_trace_full(total_time)
        elif em_free is True:
            motions = [None for _ in range(n_frames)]
            
        if len(motions) != n_frames:
            print(f'[Warning] mismatch frame count in motion calculation. Expect {n_frames}, Output {len(motions)}. Continue with {n_frames} in total') 

        from ORG.utils import static_osl, interference

        if exp_type == 'static':
            osl_nm_t = static_osl(self.mosaic['osl_um'], n_frames, std=self.cone_reflectance_parameters['static_jitter_std'])

            if coherency is False:
                reflectance = interference(osl_nm_t, wavelength_nm=self.myAOSLO.parameters['wavelength_m']*1e9)
            elif coherency is True:
                reflectance = interference(osl_nm_t, wavelength_nm=self.myAOSLO.parameters['wavelength_m']*1e9, spectral_width_nm=self.myAOSLO.parameters['spectral_bandwidth_nm'], coherency=coherency)

        return motions, reflectance
    

