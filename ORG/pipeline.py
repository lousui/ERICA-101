'''
pipelinecone_specific_osl
---------------
Pipeline for simulating ERICA images with outer segment induced cone reflectance change, using ISETBio mosaic.
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
            import mat73
            mosaic_number = self.myAOSLO.parameters['mosaic_number']
            mosaic_folder = self.myAOSLO.parameters['mosaic_folder']
            mosaic_file = os.path.normpath(os.path.join(mosaic_folder, f'mosaic_{mosaic_number:02d}.mat'))
            mosaic = mat73.loadmat(mosaic_file)
            self.mosaic = mosaic['mosaic']
            print('Unscaled mosaic grid file: %s'%mosaic_file)

            # overwrite params
            self.myAOSLO.parameters['microns_per_degree'] = self.mosaic['microns_per_deg'] 
            # self.myAOSLO.parameters['eccentricity_y_x'] = self.mosaic.ecc_deg_y_x

        self.eccentricity = self.myAOSLO.parameters["eccentricity_y_x"]        
        
    def isetbio_mosaic(self):
        '''
        Generate retina from ISETbio mosaic. 
        Analogous to ``ERICA.generate_retina.retina()``.
        Adapted from villmaria/ISETBioImageGeneration/ERICAMethods/RandomAnglesDrift_ISETBioMosaic_ForwardBackward.py
        
        Reads
        -----
        - self.mosaic : ISETbio mosaic
        - self.myAOSLO

        Modifies
        --------
        - self.retina_window_isetbio
        '''
        n_cones = self.mosaic['cone_types'].shape[0]

        # Generate rotation angles
        rotation_angles = np.random.uniform(0, 2*np.pi, size=(n_cones))

        # Two beam interference reflectance
        if self.cone_reflectance_parameters['distribution'] == 'interference':
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

        if self.cone_reflectance_parameters['renewal'] is True:
            from ORG.utils import osl_natural
            osl_renewalled = osl_natural(start_osl_um=self.mosaic['osl_um'],
                                         start_time=self.cone_reflectance_parameters['start_time'],
                                         interval_h=self.cone_reflectance_parameters['interval_h'],
                                         seed=self.cone_reflectance_parameters['seed'])
            self.mosaic['osl_um'] = osl_renewalled

    def make_blink(self):
        blink_window = self.retina_window_isetbio.copy()
        blink_window[5] = np.zeros((blink_window[5].shape))
        return blink_window

        
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
                         ):
        '''
        Add Poisson noise and Gaussian white noise to data stream.
        '''
        if noise_mean is None:
            noise_mean = self.myAOSLO.parameters['noise_mean_normalised_graylevels']
        if noise_stdev is None:
            noise_stdev = self.myAOSLO.parameters['noise_stdev_normalised_graylevels']
        if vals is None:
            vals = self.myAOSLO.parameters['poisson_scale']

        # shot noise, adapted from skimage
        # vals = len(np.unique(image_stream))
        # vals = 2 ** np.ceil(np.log2(vals))
        noisy_stream = np.random.poisson(image_stream * vals) / float(vals)
        # noisy_stream = np.random.poisson(image_stream)    # shot noise

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

        # add gaussian noise before desinusoid
        if noise == True:
            intensity = self.add_noise2stream(intensity, noise_mean, noise_stdev)

        fr_image = intensity.reshape((self.myAOSLO.parameters['number_scan_lines_including_flyback'],self.myAOSLO.parameters['number_pixels_per_line']*2))

        image = fr_image[:,:self.myAOSLO.parameters['number_pixels_per_line']]

        image_desinusoid = self.myAOSLO.desinusoid(image)

        nlines_flyback = self.myAOSLO.parameters['number_scan_lines_including_flyback'] - self.myAOSLO.parameters['number_scan_lines_excluding_flyback']

        image_desinusoid = image_desinusoid[:-nlines_flyback]

        image_dffr = self.diffraction_limited(image_desinusoid)

        if clip == True:
            saturation_val = np.percentile(image_dffr, (1.0-self.myAOSLO.parameters['proportion_saturated_pixels'])*100)
            image_dffr = np.clip(image_dffr, 0.0, saturation_val)
        
        lo, hi = self.myAOSLO.parameters['mapping_low_high'] 
        image_dffr = np.clip((image_dffr - lo) / (hi - lo), 0, 1)

        if return_intensity is not None and return_cones is None:
            return image_dffr, intensity
        
        if return_intensity is None and return_cones is not None:
            return image_dffr, cones_scanned, groundtruth_centroids
        
        if return_intensity is not None and return_cones is not None:
            return image_dffr, intensity, cones_scanned, groundtruth_centroids

        return image_dffr
    
    def make_montage(self, retina_window=None, eccentricity=None, field_of_view=None, noise:bool=True):
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

    

    def drift_trace(self, 
                    microsaccade_amplitude=None, 
                    total_time=None):
        '''
        Copied from villmaria/ISETBioImageGeneration/ERICAMethods/RandomAnglesDrift_ISETBioMosaic_ForwardBackward.py

        Generates a drift motion trace for constructing a single aoslo frame. If specified total_time, then for whole duration and will need to split time.
        
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
        if microsaccade_amplitude is None:
            microsaccade_amplitude = np.random.uniform(0, 30)
        # microsaccade_amplitude = 0 
        # Make it a nice round number :-)
        microsaccade_amplitude = round(microsaccade_amplitude, 2)

        if total_time is None:
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
    
    def em_trace_full(self, total_time, drift_only=False):
        '''
        Simulate and segmentate a motion trace with the given duration
        
        :param self: Description
        :param total_time: Description
        '''

        if drift_only is True:
            motion_yx = self.drift_trace(microsaccade_amplitude=0, total_time=total_time)
            motion_x = motion_yx[1]
            motion_y = motion_yx[0]
        else: 
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
    
    def make_trial(self, 
                   condition=0, 
                   total_time=None, 
                   save:bool=True, 
                   flush:bool=False):

        if total_time is None:
            total_time = self.cone_reflectance_parameters['duration_s']

        motions, reflectance = self.prepare_trial_params(condition, total_time)
        n_frames = len(motions)

        if self.cone_reflectance_parameters['em_free'] is True:
            images_em_free = []

        images = []
        for frnum, motion in enumerate(motions):
            if flush:
                sys.stdout.write('Generating image %03i of %03i\r'%(frnum+1, n_frames))
                sys.stdout.flush()
            retina_window = self.retina_window_isetbio.copy()
            retina_window[5] = reflectance[:,frnum]
            image = self.make_aoslo_image(retina_window, motion)
            images.append(image)

            if self.cone_reflectance_parameters['em_free'] is True:
                image = self.make_aoslo_image(retina_window, eye_motion_arcmin=None)
                images_em_free.append(image)

        # return motions, images, osl_nm_t

        if save == True:
            from ORG.utils import next_filename
            data_dir = './experiments'
            os.makedirs(data_dir, exist_ok=True)
            labels = {0: 'A', 1:'L', 2:'M', 3:'S'}
            prefix = f"{self.cone_reflectance_parameters['exp_type']}_{labels[condition]}_"
            suffix = '.npy'

            fnamef, fname = next_filename(data_dir, prefix, suffix)
            np.save(os.path.join(data_dir, fname+'_motions.npy'), np.asarray(motions))
            # print(f"Motions saved to file: {os.path.join(data_dir, fname+'_em_trace.npy')}")

            np.save(os.path.join(data_dir, fnamef), np.asarray(images))
            # print(f'Save to file: {os.path.join(data_dir, fnamef)}')
    
    def prepare_trial_params(self, 
                             condition=0, 
                             total_time=None, 
                             em_free=False, 
                             coherency=False,
                             drift_only=None):

        if total_time is None:
            total_time = self.cone_reflectance_parameters['duration_s']
        
        if condition == 4:
            exp_type = 'static'
        else:
            exp_type = 'flicker'

        # print(exp_type)

        frame_time = self.myAOSLO.n_samples * self.myAOSLO.time_resolution
        n_frames = int(total_time / frame_time)

        if em_free is False:
            motions = self.em_trace_full(total_time, drift_only=drift_only)
        elif em_free is True:
            motions = [None for _ in range(n_frames)]
            
        if len(motions) != n_frames:
            print(f'[Warning] mismatch frame count in motion calculation. Expect {n_frames}, Output {len(motions)}. Continue with {n_frames} in total') 
       

        from ORG.utils import static_osl, flicker_osl, dark_flash_opd, interference, cone_specific_contrast, condition_contrast

        if exp_type == 'static':
            osl_nm_t = static_osl(self.mosaic['osl_um'], n_frames, std=self.cone_reflectance_parameters['static_jitter_std'])

            if coherency is False:
                reflectance = interference(osl_nm_t, wavelength_nm=self.myAOSLO.parameters['wavelength_m']*1e9)
            elif coherency is True:
                reflectance = interference(osl_nm_t, wavelength_nm=self.myAOSLO.parameters['wavelength_m']*1e9, spectral_width_nm=self.myAOSLO.parameters['spectral_bandwidth_nm'], coherency=coherency)

        elif exp_type == 'flicker':
            LMS_contrast = condition_contrast(condition, self.cone_reflectance_parameters)
            contrast_map = cone_specific_contrast(self.mosaic['cone_types'], LMS_contrast)
            osl_nm_t = flicker_osl(self.mosaic['osl_um'],
                                   duration=total_time,
                                   n_frames=n_frames,
                                   frequency=self.cone_reflectance_parameters['frequency_Hz'],
                                   max_amp_nm=self.cone_reflectance_parameters['max_amp_nm'], 
                                   std=self.cone_reflectance_parameters['static_jitter_std'],
                                   contrast_map=contrast_map)
            
            if self.cone_reflectance_parameters['pre_stim_nframes'] != 0:
                pre_stim_osl_nm_t = static_osl(self.mosaic['osl_um'], 
                                            n_frames = self.cone_reflectance_parameters['pre_stim_nframes'],
                                            std = self.cone_reflectance_parameters['static_jitter_std'])
                osl_nm_t = np.concatenate( (pre_stim_osl_nm_t, osl_nm_t), axis=1)
            if coherency is False:
                reflectance = interference(osl_nm_t, wavelength_nm=self.myAOSLO.parameters['wavelength_m']*1e9)
            elif coherency is True:
                reflectance = interference(osl_nm_t, wavelength_nm=self.myAOSLO.parameters['wavelength_m']*1e9, spectral_width_nm=self.myAOSLO.parameters['spectral_bandwidth_nm'], coherency=coherency)

        elif exp_type == 'd-flash':
            # right now only 8% bleach achromatic is implemented
            t_sequence = np.arange(n_frames) * frame_time - self.cone_reflectance_parameters['pre_stim_s']
            opd_nm = dark_flash_opd(t=t_sequence, l0_nm=self.mosaic['osl_um'])

            if coherency is False:
                reflectance = interference(opd_nm, wavelength_nm=self.myAOSLO.parameters['wavelength_m']*1e9)
            elif coherency is True:
                reflectance = interference(opd_nm, wavelength_nm=self.myAOSLO.parameters['wavelength_m']*1e9, spectral_width_nm=self.myAOSLO.parameters['spectral_bandwidth_nm'], coherency=coherency)
            
        

        return motions, reflectance
    

