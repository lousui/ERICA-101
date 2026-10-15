'''
utils.py
---------------
Utilities used in pipeline.py

Sirui Liu, Mar 2026
'''

import numpy as np
from ERICA import peak_detection

def to_python(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_python(v) for v in obj]
    return obj

def write_grayscale_avi(arr: np.ndarray, path: str, fps: int = 30):
        import cv2
        """
        arr: either
        - (T, H, W) video frames, or
        - (H, W) single image (saved as a 1-frame video)
        Values should be uint8 in [0, 255].
        """
        arr = np.asarray(arr)
        arr = ((arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255).astype(np.uint8)

        if arr.ndim == 2:
            arr = arr[None, ...]  # make it 1-frame video
        elif arr.ndim != 3:
            raise ValueError("Expected array shape (T, H, W) or (H, W)")

        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)

        t, h, w = arr.shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  
        writer = cv2.VideoWriter(path, fourcc, fps, (w, h), isColor=False)

        if not writer.isOpened():
            raise RuntimeError("Could not open video writer")

        for frame in arr:
            writer.write(frame)  # single-channel grayscale frame

        writer.release()
        return

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


def static_osl(start_osl_um, n_frames, std=0.1):
    '''
    Add random fluctuation of outer segment length under control condition.

    Parameters:
        start_osl_um (ndarray): An array of outer segment length in microns.
        n_frames (int): Number of frames for the given duration.
        std (float): Standard deviation for random fluctuation of outer segment length.
    
    Returns:
        osl_nm (ndarray): An array of [n_cones, n_frames] of outer segment length in nm. 
    '''
    n_cones = start_osl_um.shape[0]
    osl_nm = start_osl_um * 1e3
    osl_jittered_nm = np.random.normal(osl_nm[:,None], std, size=(n_cones, n_frames))
    return osl_jittered_nm

def interference(
        osl_nm_t=None, 
        opd_nm=None,
        wavelength_nm=830, 
        spectral_width_nm=None, 
        coherency=False,
        ):
    '''
    Two beam interference of the given outer segment length traces. 
    
    Parameters
    ----------
    - osl_nm_t : ndarray
        An array of [n_cones, n_frames] of outer segment length traces in nm.
    - opd_nm : ndarray
        An array of [n_cones, n_frames] of optical path length in nm, doesn't 
        need to pass stuff in if you have passed `osl_nm_t`.
    - wavelength_nm : float
        Imaging lightsource wavelength in nm.
    - spectral_width_nm : float
        Spectral width of the lightsource in nm, for degree of coherency.
    - coherency : bool
        Whether or not take degree of coherency into account
    
    Returns
    --------
    - reflectance : ndarray
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

