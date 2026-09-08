import numpy as np

from ERICA import peak_detection

def assign_cone_types(cone_locations_um, cone_ratios=[0.6, 0.3, 0.1], microns_per_degree=297.0, s_free_radius_deg=0.15, seed=None):
    '''
    Assign cone types (L, M, S) to cone locations.
    Adapted from isetbio/isettools/cones/cmosaic/@cMosaic/assignConeTypes.m

    Parameters
    ----------
    cone_locations_um : ndarray
        An array containing the locations of the cones, typically the output
        of the reaction diffusion model, scaled by a particular retinal
        eccentricity.
    cone_ratios : ndarray
        [L, M, S] cone ratios
    microns_per_degree : float
        The number of microns per degree on the retinal surface - stored in the
        system parameters file. Default = 297.0
    s_free_radius_deg : float
        Eccentricity in degree define S-cones-free zone at the fovea

    Returns
    -------
    cone_types : ndarray
        Array where values indivate cone type (1=L, 2=M, 3=S)
    '''
    
    if seed is not None:
        np.random.seed(seed)
    else:
        np.random.seed(np.random.randint(0,99999))
    
    l_cone_id, m_cone_id, s_cone_id = 1, 2, 3
    
    n_cones = cone_locations_um.shape[1]
    cone_types = np.zeros((n_cones,1), dtype=int)

    if sum(cone_ratios) == 0:
        raise ValueError('cone_ratios cannot be summed to zero')
    else:
        cone_ratios = np.array(cone_ratios) / sum(cone_ratios) # normalise to sum to 1

    # eccentricities of all cones
    eccentricities = np.sqrt(cone_locations_um[0]**2 + cone_locations_um[1]**2) / microns_per_degree

    distances, _ = peak_detection.findNeighbours(np.transpose(cone_locations_um), number_neighbours=6)
    # Take the median distance between a cone and its 3 nearest neighbours
    cone_spacings = np.median(distances[:,1:4], axis=1)

    # deal with s-cone free zone
    foveal_mask = eccentricities <= float(s_free_radius_deg)
    foveal_indices = np.nonzero(foveal_mask)[0]
    peripheral_indices = np.setdiff1d(np.arange(n_cones), foveal_indices)

    desired_s_num = int(round(cone_ratios[s_cone_id-1] * peripheral_indices.size))
    
    # choose peripheral LM based on S-cone regularity
    if (cone_ratios[s_cone_id-1] > 0) and (cone_ratios[s_cone_id-1] <= 0.5):
        peripheral_pos = cone_locations_um[:,peripheral_indices]
        peripheral_spacing = cone_spacings[peripheral_indices]
        lm_mask_peripheral = _determine_lm_indicies(peripheral_pos, peripheral_spacing, cone_ratios[s_cone_id-1], desired_s_num)
        lm_indices_peripheral = peripheral_indices[np.nonzero(lm_mask_peripheral)[0]]
    else: # randomly assign
        perm = np.random.permutation(peripheral_indices.size)
        lm_num = n_cones - desired_s_num
        lm_indices_peripheral = peripheral_indices[perm[:lm_num]]
    
    lm_indices_all = np.unique(np.concatenate([foveal_indices, lm_indices_peripheral]).astype(int))
    lm_ratio = cone_ratios[l_cone_id-1]/[m_cone_id-1]

    # randomly assign
    p = np.random.rand(lm_indices_all.size)
    if np.isinf(lm_ratio): # no M
        choose_local = np.arange(lm_indices_all.size)
    elif lm_ratio == 0:  # no L
        choose_local = np.array([], dtype=int)
    else:
        cutoff = lm_ratio / (1.0 + lm_ratio)
        choose_local = np.nonzero(p < cutoff)[0]
    
    l_indices = lm_indices_all[choose_local]
    m_indices = np.setdiff1d(lm_indices_all, l_indices)
    s_indices = np.setdiff1d(np.arange(n_cones), lm_indices_all)

    cone_types = np.zeros(n_cones, dtype=int)
    cone_types[l_indices] = l_cone_id
    cone_types[m_indices] = m_cone_id
    cone_types[s_indices] = s_cone_id
    return cone_types

def _determine_lm_indicies(cone_location_deg, cone_spacings, desired_s_ratio, desired_s_num):
    '''
    Helper function for placing LM in quasi-regular S mosaics
    '''
    n_cones = cone_location_deg.shape[1]

    # determine relative s-cone spacing from s-cone density
    test_s_spacing = np.arange(1.1, 10.01, 0.01)
    s_density_from_norm_s_spacing = 0.01 * (1.0 + 30.0 * np.exp(-0.8 * (test_s_spacing - 1.0) ** 0.9))
    idx = np.argmin(np.abs(desired_s_ratio - s_density_from_norm_s_spacing))
    relative_s_spacing = test_s_spacing[idx]

    # eccentricities of all cones
    eccentricities = np.sqrt(np.sum(np.transpose(cone_location_deg)**2, axis=1))

    # distances with 200 nearest neighbours
    distances, indicies = peak_detection.findNeighbours(np.transpose(cone_location_deg), number_neighbours=200)
    # exclude itself
    distances = distances[:, 1:]
    neighbours = indicies[:, 1:]

    # assign s cones with spacing greater than relative_s_spacing
    remaining = set(range(n_cones))
    s_indices = []
    lm_indices = []

    if remaining:
        # current = int(np.argmin(eccentricities))
        current = min(remaining, key=lambda i:eccentricities[i])
    else:
        return np.ones(n_cones, dtype=bool)
    
    while remaining:
        s_indices.append(current)
        current_exclusion_radius = cone_spacings[current] * relative_s_spacing
        nearby_indices = np.nonzero(distances[current, :]<current_exclusion_radius)[0]
        cone_indices = neighbours[current, nearby_indices].tolist()
        lm_indices.extend(cone_indices)
        # update remaining
        remaining.difference_update(s_indices)
        remaining.difference_update(lm_indices)
        if not remaining:
            break
        current = min(remaining, key=lambda i:eccentricities[i])

    lm_indices = np.unique(np.array(lm_indices, dtype=int))
    s_indices = np.array(s_indices, dtype=int)

    # adjust s count to match desired_s_num
    if s_indices.size > desired_s_num:
        # random sample
        perm = np.random.permutation(s_indices.size)
        s_chosen = s_indices[perm[:desired_s_num]]
        # lm_indices = np.setdiff1d(np.arrange(n_cones, dtype=int), s_indices)
        lm_mask = np.ones(n_cones, dtype=bool)
        lm_mask[s_chosen] = False
    elif s_indices.size < desired_s_num:
        lm_candidates = np.setdiff1d(np.arange(n_cones), s_indices)
        perm = np.random.permutation(lm_candidates.size)
        desired_lm_num = n_cones - desired_s_num
        lm_chosen = lm_candidates[perm[:desired_lm_num]]
        lm_mask = np.zeros(n_cones, dtype=bool)
        lm_mask[lm_chosen] = True
    else:
        lm_mask = np.ones(n_cones, dtype=bool)
        lm_mask[s_indices] = False
    return lm_mask

def generate_cone_types_osl(cone_types, lcones_osl_um_mean=25.0, lcones_osl_um_std=3.333, mcones_osl_um_mean=25.0, mcones_osl_um_std=3.333, scones_osl_um_mean=22.0, scones_osl_um_std=1.667, osl_min_um=15.0, osl_max_um=35.0):

    # TODO - Now is limited to 3-4 deg eccentricity, extend range to different eccentricity. Fovea cones have longer outer segments.

    '''
    Create an array of outer segment map based on cone types (L, M, S). 
    S cones typically have shorter outer segments.
    Outer segment length is eccentricity-dependent.
    Default parameters are from DOI: 10.1117/12.2650609
    
    Parameters
    ----------
    cone_types : ndarray
        Array where values indicate cone type (1=L, 2=M, 3=S)
    lcones_osl_um_mean : float
        Mean outer segment length for L-cones (default 25.0um)
    lcones_osl_um_std : float
        Standard deviation of outer segment length for L-cones (default 3.333um)
    mcones_osl_um_mean : float
        Mean outer segment length for M-cones (default 25.0um)
    mcones_osl_um_std : float
        Standard deviation of outer segment length for M-cones (default 3.333um)
    scones_osl_um_mean : float
        Mean outer segment length for S-cones (default 22.0um)
    scones_osl_um_std : float
        Standard deviation of outer segment length for S-cones (default 1.667um)
    osl_min_um : float
        Lower bound of outer segment length (default 15.0um)
    osl_max_um : float
        Upper bound of outer segment length (default 35.0um)
        
    Returns
    -------
    cone_specific_osl : ndarray
        Array of outer segment lengths
    '''
    cone_specific_osl = np.zeros(cone_types.shape)

    # For L cones
    l_mask = cone_types == 1
    n_lcones = np.sum(l_mask)
    if n_lcones > 0:
        cone_specific_osl[l_mask] = np.random.normal(lcones_osl_um_mean, lcones_osl_um_std, size=n_lcones)

    # For M cones
    m_mask = cone_types == 2
    n_mcones = np.sum(m_mask)
    if n_mcones > 0:
        cone_specific_osl[m_mask] = np.random.normal(mcones_osl_um_mean, mcones_osl_um_std, size=n_mcones)

    # For S cones
    s_mask = cone_types == 3
    n_scones = np.sum(s_mask)
    if n_scones > 0:
        cone_specific_osl[s_mask] = np.random.normal(scones_osl_um_mean, scones_osl_um_std, size=n_scones)
    
    cone_specific_osl = np.clip(cone_specific_osl, a_min=osl_min_um, a_max=osl_max_um)

    return cone_specific_osl


def shedding_probability(start_time: str, interval_h: float):
    '''
    Shedding event probability at different time. 
    Assume 16/8 awake/sleep cycle. Parameters taken from DOI: 10.1364/BOE.7.004554
    - morning: 8:00-11:59
    - afternoon: 12:00-17:59
    - evening: 18:00-23:59
    - sleep: 0:00-7:59

    Parameter
    ---------
    start_time : string
        The start time in 24h-system, e.g. Half pass six, pm, is 18:30
    interval_h : float
        The time interval in hour, e.g. 1 hr 30 min is 1.5
    
    Return
    ------
    shed_probs : ndarray
        Array of shedding probability(s) based on the window span
    '''

    shed_probs = [
        0.143,  # morning
        0.057,  # afternoon
        0.040,  # evening
        0.054   # sleep -- hypothetical average
    ]
    shed_probs = np.array(shed_probs)

    bools, order = _touched_windows(start_time, interval_h)

    return shed_probs[bools]
        

def _touched_windows(start_time: str, interval_h: float):
    """
    Returns
    -------
    bools : list 
        [True/False] for [morning, afternoon, evening, sleep].
    touched_wins : list 
        List of window names touched in chronological order.
    """

    names = ["morning", "afternoon", "evening", "sleep"]
    wmins = [
        (8*60, 12*60),    # morning
        (12*60, 18*60),   # afternoon
        (18*60, 24*60),   # evening
        (0, 8*60)         # sleep
    ]

    def _to_minutes(hm):
        h, m = map(int, hm.split(":"))
        return h*60 + m

    start = _to_minutes(start_time)
    dur = int(round(interval_h * 60))
    # if covers full day or more -> all windows touched
    if dur % (24*60) == 0 and dur >= 24*60:
        return [True]*4, names.copy()

    end = start + dur  # may be >1440
    # produce up to two segments in 0..1440 to test overlaps
    segments = []
    if end <= 1440:
        segments.append((start, end))
    else:
        segments.append((start, 1440))
        segments.append((0, end % 1440))

    bools = [False]*4
    touched_wins = []
    seen = set()
    for seg_s, seg_e in segments:
        for i, (ws, we) in enumerate(wmins):
            if seg_s < we and seg_e > ws:   # overlap test
                bools[i] = True
                if names[i] not in seen:
                    touched_wins.append(names[i])
                    seen.add(names[i])

    return bools, touched_wins

def osl_natural(start_osl_um, start_time, interval_h, renewal_rates_nm_per_hr=(93,113),seed=None):
    # TODO: add over 24h limit
    '''
    Simulate outer segment lengths after a time interval.
    Assumptions:
    - Outer segment renewal is a linear, constant process.
    - During the given time interval no more than one disc shedding event happen per cone
    The default outer segment renewal parameters taken from DOI: 10.1364/OE.18.005257, measured at 1.8-deg eccentricity. 
    Disc shedding relavent default parameters taken from DOI: 10.1364/BOE.7.004554, measured at 2-deg and 3-deg eccentricity
    
    Parameters
    ----------
    start_osl_um : ndarray
        The outer segment lengths array in micron at the give start time.
    start_time : string
        The start time in 24h-system, e.g. Half pass six, pm, is 18:30
    interval_h : float
        The time interval in hour, e.g. 1 hr 30 min is 1.5
    renewal_rates_nm_per_hr : ndarray
        The range of outer segment renewal rate specified in (lower_rate, upper_rate). 
    shed_length_range_um : ndarray
        The range of outer segment length shorten due to disc shedding,
        specified in (lower_bound, upper_bound). 
    seed : int
        Random seed for reproducibility.
    
    Returns
    -------
    osl_later : ndarray
        The outer segment lengths array after a period of time.
    '''
    if seed is not None:
        np.random.seed(seed)
    else:
        np.random.seed(np.random.randint(0, 99999))

    n_cones = start_osl_um.size

    # randomly sample renewal rates - uniform distribution
    renewal_rates = np.random.uniform(renewal_rates_nm_per_hr[0], renewal_rates_nm_per_hr[1], size=n_cones)
    renewal_rates = renewal_rates*1e-3  # convert to um

    # get disc shedding probability
    shed_probs = shedding_probability(start_time, interval_h)

    osl_renewalled = start_osl_um + renewal_rates*interval_h
    
    n_cones_shed = (n_cones * shed_probs).astype(int)     # number of cones exhibit shed events
    
    # assume one cone shed once
    perm = np.random.permutation(n_cones)
    shed_cones_all_indices = perm[:n_cones_shed.sum()]

    # length loss due to disc shedding - beta distribution
    alpha = 2
    beta = 5
    scale = 6  # um, upper bound
    length_loss = scale * np.random.beta(alpha, beta, size=shed_cones_all_indices.size)
    
    osl_shedded = osl_renewalled[shed_cones_all_indices] - length_loss
    osl_renewalled[shed_cones_all_indices] = osl_shedded

    return osl_renewalled

def generate_drift(n_samples, total_displacement):
    """
    Copied from villmaria/ISETBioImageGeneration/ERICAMethods/DriftSimulation.py
    Generates a drift signal with a specified number of samples and total displacement.
    
    Parameters:
        n_samples (int): Number of samples in the drift signal.
        total_displacement (float): Total displacement of the drift signal.
        
    Returns:
        np.ndarray: Drift signal.

    https://elifesciences.org/articles/98648#content
    Figure 3c

    """

    # Generate a random walk
    # n samples calculated from 
    mean = 1 # Mean step size
    std = 0.15 # How far to deviate from the mean

    random_walk = np.random.normal(loc=mean, scale=std, size=n_samples)
    # print(f'Random Walk Size    : {random_walk.size}')
    # print(f'Random Walk Shape   : {random_walk.shape}')
    # print(f"Mean of random walk: {np.mean(random_walk)}")
    # print(f"Standard deviation of random walk: {np.std(random_walk)}")

    # print(f"Sum of random walk: {np.sum(random_walk)}")
    # Normalize the random walk to match the total displacemen
    # random_walk / (np.sum(random_walk) * total_displacement)
    random_walk = np.divide(random_walk, np.sum(random_walk) *total_displacement)

    # Random directions
    angles = np.random.uniform(0, np.pi*2, size=n_samples)
    # Compute dx, dy
    dx = random_walk * np.cos(angles)
    dy = random_walk * np.sin(angles)

    # Cumulative positions
    x = np.cumsum(dx)
    y = np.cumsum(dy)

    positions = np.zeros((2, n_samples))
    positions[1] = x
    positions[0] = y

    # print to total drfit 
    # print(f"Total drift: {np.sum(random_walk)}")

    return positions

def static_osl(start_osl_um, n_frames, std=0.01):
    n_cones = start_osl_um.shape[0]
    osl_nm = start_osl_um * 1e3
    osl_jittered_nm = np.random.normal(osl_nm[:,None], std, size=(n_cones, n_frames))
    return osl_jittered_nm
    
def flicker_osl(start_osl_um, duration, n_frames, frequency, max_amp_nm, std, contrast_map):
    '''
    Generate outer segment length trace response to flicker
    
    Parameters
    ----------
    - start_osl_um : ndarray
        An array of outer segment length in microns.
    - duration : float
        Time duration of the trace.
    - n_frames : int
        Number of frames for the given duration.
    - frequency : float
        Flicker frequency.
    - max_amp_nm : float
        Maximum modulation amplitude of outer segment length.
    - std : float
        Standard deviation for random fluctuation of outer segment length.
    - contrast_map : ndarray
        cone contrast map

    Returns
    -------
    - osl_nm: ndarray
        An array of [n_cones, n_frames] of outer segment length in nm. 
    '''
    omega = 2 * np.pi * frequency
    timepoints = np.linspace(0, duration, n_frames)
    # osl_nm = np.repeat(start_osl_um[:, None]*1e3, n_frames, axis=1)
    osl_nm = static_osl(start_osl_um, n_frames, std)
    # osl_nm = osl_nm + max_amp_nm * np.sin(omega*timepoints)
    max_amp_map = max_amp_nm * contrast_map
    max_amp_map = max_amp_map.reshape(max_amp_map.shape[0], 1)
    osl_nm = osl_nm + max_amp_map * np.sin(omega*timepoints).reshape(1, n_frames)
    return osl_nm

def dark_flash_opd(t, 
                   A0 = -36.4, 
                   A1 = 200.8, 
                   tau_a = 9.5, 
                   tau_b = 0.07, 
                   l0_nm = 25*1e3, 
                   n = 1.43,
                   ):
    ''' 
    Dark adapted flash optical path length difference
    Model from https://www.science.org/doi/full/10.1126/sciadv.adq7332

    default values 8% cone bleach
    A0 = -36.4 nm       -- fast constraction
    A1 = 200.8 nm       -- swelling amplitude
    tau_a = 9.5 s^-1    -- swelling rate parameter
    tau_b = 0.07 s^-1   -- late contraction rate parameter

    l0_nm = 25 um      -- starting outer segment length
    n  = 1.43          -- outer segment refractive index
    '''
    if type(l0_nm) == float:
        l0_nm = np.asarray([l0_nm])
    elif type(l0_nm) == list:
        l0_nm = np.asarray(l0_nm)

    l0_nm = np.atleast_1d(l0_nm)

    u = np.heaviside(t, 1)
    delta_opd = u * (A0 + A1 * (-np.exp(-tau_a * t) + np.exp(-tau_b * t)))

    opd_nm = 2 * n * l0_nm[:, None] + delta_opd[None, :]
    return opd_nm

def interference(
        osl_nm_t=None, opd_nm=None,
        wavelength_nm=830, 
        spectral_width_nm=None, 
        coherency=False,
        ):
    '''
    Two beam interference of the given outer segment length traces. 
    
    Parameters
    ----------
    - osl_nm_t
        An array of [n_cones, n_frames] of outer segment length traces in nm.
    - wavelength_nm
        Imaging lightsource wavelength in nm.
    - spectral_width_nm
        Spectral width of the lightsource in nm, for degree of coherency.
    - coherency
        Whether or not take degree of coherency into account
    
    Returens
    --------
    - reflectance
        An array of [n_cones, n_frames] of cone reflectance.
    '''

    if opd_nm is None and osl_nm_t is not None:
        NO = 1.430
        opd_nm = 2 * NO * osl_nm_t

    d_phi = 2 * np.pi * opd_nm / wavelength_nm

    if coherency is False:
        reflectance = 0.5 * (1 + np.cos(d_phi)) 
    else:
        coherence_length_nm = 2 * np.log(2) * wavelength_nm**2 / (NO * np.pi * spectral_width_nm)
        deg_of_coherency = np.abs(np.exp(-(opd_nm/(coherence_length_nm*1e9))**2))
        reflectance = 0.5 * (1 + deg_of_coherency * np.cos(d_phi)) 

    return reflectance

def cone_specific_contrast(cone_types, LMS_contrast):

    contrast_map = np.zeros(cone_types.shape)

    l_mask = cone_types == 1
    n_lcones = np.sum(l_mask)
    if n_lcones > 0:
        contrast_map[l_mask] = LMS_contrast[0]

    # For M cones
    m_mask = cone_types == 2
    n_mcones = np.sum(m_mask)
    if n_mcones > 0:
        contrast_map[m_mask] = LMS_contrast[1]

    # For S cones
    s_mask = cone_types == 3
    n_scones = np.sum(s_mask)
    if n_scones > 0:
        contrast_map[s_mask] = LMS_contrast[2]
    
    return contrast_map

def condition_contrast(condition_idx, params):
    if condition_idx == 0:
        LMS_contrast = params['Achr_contrast']
    elif condition_idx == 1:
        LMS_contrast = params['L_iso_contrast']
    elif condition_idx == 2:
        LMS_contrast = params['M_iso_contrast']
    elif condition_idx == 3:
        LMS_contrast = params['S_iso_contrast']
    else:
        raise ValueError(f'Condition index must be 0 [Achromatic], 1 [L isolating], 2 [M isolating], or 3 [S isolating], while the input condition index is {condition_idx}')
    return LMS_contrast

# def next_filename(folder, prefix, suffix, digits=3):
#     '''
#     Docstring for next_filename
    
#     :param folder: Description
#     :param prefix: Description
#     :param suffix: Description
#     :param digits: Description

#     Returns
#     -------
#     - str: Full file name with suffix
#     - str: File name without suffix

#     '''
#     from pathlib import Path
#     import re

#     folder = Path(folder)
#     pattern = re.compile(rf"{prefix}(\d+){suffix}")
    
#     nums = []
#     for f in folder.iterdir():
#         match = pattern.fullmatch(f.name)
#         if match:
#             nums.append(int(match.group(1)))
#     next_num = max(nums) + 1 if nums else 1
#     return f"{prefix}{next_num:0{digits}d}{suffix}", f"{prefix}{next_num:0{digits}d}"

def next_filename(folder, prefix, suffix="", digits=3, kind="folder"):
    from pathlib import Path
    import re

    folder = Path(folder)

    base_pattern = re.compile(rf"^{re.escape(prefix)}$")
    numbered_pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)$")

    nums = []

    for p in folder.iterdir():
        if kind == "folder" and not p.is_dir():
            continue
        if kind == "file" and not p.is_file():
            continue

        name = p.name

        if base_pattern.fullmatch(name):
            nums.append(0)
        else:
            m = numbered_pattern.fullmatch(name)
            if m:
                nums.append(int(m.group(1)))

    next_num = max(nums) + 1 if nums else 1
    base = f"{prefix}_{next_num:0{digits}d}"

    return base + suffix, base

