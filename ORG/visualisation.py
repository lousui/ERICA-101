import matplotlib.pyplot as plt
import numpy

def plot_cone_spacing(mapped, microns_per_degree=297.0, fnum=1, n_neighbours=3):
    from ERICA import generate_retina

    # Detect the spacings between each cone and its nearest neighbours
    def ensure_nx2(arr):
        arr = numpy.asarray(arr)

        if arr.ndim != 2:
            raise ValueError("Input must be a 2D array")

        rows, cols = arr.shape

        if cols == 2:
            # already n x 2
            return arr
        elif rows == 2:
            # 2 x n → transpose
            return arr.T
        else:
            raise ValueError("Input must be of shape (n, 2) or (2, n)")
    
    mapped = ensure_nx2(mapped)
    print(mapped.shape)
    spacing = generate_retina.determineSpacing(mapped, n_neighbours=n_neighbours)
   
    # Calculate the eccentricities (in microns) of each cone cell
    eccentricies_points_um = numpy.sqrt(mapped[:,1]**2+mapped[:,0]**2)
    print(eccentricies_points_um.shape)

    # Calculate the expected cone spacings based on Curcio et al. (1990) data
    radial = numpy.linspace(0,10,20)
    y = numpy.zeros(radial.shape)
    eccentricities_compare = numpy.array([y,radial])
    spacing_compare = generate_retina.spacingFromEccentricity(eccentricities_compare)

    # Plot the expected cone spacings in black and show the spacings of the generated mosaic of cones in blue
    fig_1 = plt.figure(fnum, figsize=(10,4))
    ax1_1 = fig_1.add_subplot(111)
    ax1_1.scatter(eccentricies_points_um/microns_per_degree, spacing, s=2, color='b', label='Generated mosaic')
    ax1_1.plot(radial,spacing_compare, color='black', label='Curcio et al. 1990 fit')
    ax1_1.scatter(generate_retina.CONE_POSITIONS,generate_retina.CONE_SPACING_UM, color='black', label='Curcio et al. 1990 data', s=6)
    
    ax1_1.set_xlim(0,radial.max())
    ax1_1.set_xlabel('Eccentricity (degrees)')
    ax1_1.set_ylabel('Cone spacing (microns)')
    plt.legend(loc='upper left', bbox_to_anchor=(1,1))
    plt.show()
    

def visualise_mosaic_grid(cone_positions_window, unit='deg', microns_per_degree=297.0):
    fig_1 = plt.figure(1, figsize=(10,10))
    ax1_1 = fig_1.add_subplot(111)

    if unit == 'micron':
        ax1_1.scatter(cone_positions_window[1], cone_positions_window[0], s=1)
    elif unit == 'deg':
        ax1_1.scatter(cone_positions_window[1]/microns_per_degree, cone_positions_window[0]/microns_per_degree, s=1) 
    else:
        raise SyntaxError('Unit must be either "micron" or "deg".')
    
    ax1_1.set_xlabel(f'Horizontal Postion ({unit})')
    ax1_1.set_ylabel(f'Vertical Postion ({unit})')

    plt.show()

def visualise_cone_mosaic(cone_location_um, cone_types, unit='deg', microns_per_degree=297.0):
    colors = {1:'red', 2:'green', 3:'blue'}
    labels = {1:'L cone', 2:'M cone', 3:'S cone'}

    def ensure_2xn(arr):
        import numpy as np
        arr = np.asarray(arr)

        if arr.ndim != 2:
            raise ValueError("Input must be a 2D array")

        rows, cols = arr.shape

        if rows == 2:
            return arr
        elif cols == 2:
            return arr.T
        else:
            raise ValueError("Input must be of shape (n, 2) or (2, n)")
    
    cone_location_um = ensure_2xn(cone_location_um)

    if unit == 'deg':
        cone_location = cone_location_um / microns_per_degree
    elif unit == 'micron':
        cone_location = cone_location_um
    else:
        raise SyntaxError('Unit must be either "micron" or "deg".')
    
    fig, ax = plt.subplots(1,1,figsize=(12,12))
    alpha_vals = [0.7, 0.4, 0.3]
    for cone_type in [1, 2, 3]:
        idx = cone_types == cone_type
        ax.scatter(cone_location[1, idx], cone_location[0, idx], s=8, c=colors[cone_type], label=labels[cone_type], edgecolors='black', linewidths=0.1, alpha=alpha_vals[cone_type-1])

    ax.set_xlabel(f'X position ({unit})')
    ax.set_ylabel(f'Y position ({unit})')
    ax.set_title('Cone Mosaic')
    ax.axis('equal')

    plt.show()

def visualise_cone_type_and_osl(cone_location_um, cone_types, osl_cone_map, unit='deg'):

    colors = {1:'red', 2:'green', 3:'blue'}
    labels = {1:'L cone', 2:'M cone', 3:'S cone'}

    def ensure_2xn(arr):
        import numpy as np
        arr = np.asarray(arr)

        if arr.ndim != 2:
            raise ValueError("Input must be a 2D array")

        rows, cols = arr.shape

        if rows == 2:
            return arr
        elif cols == 2:
            return arr.T
        else:
            raise ValueError("Input must be of shape (n, 2) or (2, n)")
    
    cone_location_um = ensure_2xn(cone_location_um)

    if unit == 'deg':
        micron_per_degree = 297.0
        cone_location = cone_location_um / micron_per_degree
    elif unit == 'micron':
        cone_location = cone_location_um
    else:
        raise SyntaxError('Unit must be either "micron" or "deg".')
    
    fig, (ax, ax1) = plt.subplots(1,2,figsize=(12,6))
    for cone_type in [1, 2, 3]:
        idx = cone_types == cone_type
        ax.scatter(cone_location[1, idx], cone_location[0, idx], s=10, c=colors[cone_type], label=labels[cone_type], edgecolors='black', linewidths=0.25, alpha=0.8)

    ax.set_xlabel(f'X position ({unit})')
    ax.set_ylabel(f'Y position ({unit})')
    ax.set_title('Cone Mosaic')
    ax.axis('equal')

    osl = ax1.scatter(cone_location[1], cone_location[0], c=osl_cone_map, cmap='viridis', s=5, edgecolors='black', linewidths=0.1, alpha=0.8)
    fig.colorbar(osl, ax=ax1)
    ax1.set_xlabel(f'X position ({unit})')
    ax1.set_ylabel(f'Y position ({unit})')
    ax1.set_title('Outer Segment Length')
    ax1.axis('equal')

    # plt.tight_layout()
    plt.show()

def visualise_two_session_imaging(image_1_diffr, image_2_diffr, image_1_noise, image_2_noise, interval, noise_mode, osl_cone_map, osl_cone_map_new):
    fig, axs = plt.subplots(3,3, figsize=(20,20))

    axs[0,0].imshow(image_1_diffr, cmap='gray')
    axs[0,0].set_title('Session 1\n(Diffraction limit)')

    axs[0,1].imshow(image_2_diffr, cmap='gray')
    axs[0,1].set_title(f'After {interval}h\n(Diffraction limit)')

    diff1 = axs[0,2].imshow(image_2_diffr-image_1_diffr, cmap='bwr')
    fig.colorbar(diff1, ax=axs[0,2])
    axs[0,2].set_title(f'Difference')

    axs[1,0].imshow(image_1_noise, cmap='gray')
    axs[1,0].set_title(f'Session 1\n(Diffraction limit, {noise_mode} noise)')

    axs[1,1].imshow(image_2_noise, cmap='gray')
    axs[1,1].set_title(f'After {interval}h\n(Diffraction limit, {noise_mode} noise)')

    diff2 = axs[1,2].imshow(image_2_noise-image_1_noise, cmap='bwr')
    axs[1,2].set_title(f'Difference')
    fig.colorbar(diff2, ax=axs[1,2])

    axs[2,0].hist(osl_cone_map)
    axs[2,0].set_title('Session 1')
    axs[2,0].set_xlabel('OSL(um)')
    axs[2,0].set_ylabel('# of Cones')

    axs[2,1].hist(osl_cone_map_new)
    axs[2,1].set_title(f'After {interval}h')
    axs[2,1].set_xlabel('OSL(um)')
    axs[2,1].set_ylabel('# of Cones')
    plt.show()


def eye_motion_1D(t, mx, my, ms_marker=[[numpy.nan, numpy.nan]], downsample_for_plotting=8000, cmap=plt.get_cmap('viridis')):
    """
    Plots the horizontal (x) and vertical (y) data as separate lines.

    Parameters:
        t (ndarray): time samples.
        mx (ndarray): motion x
        my (ndarray): motion y
        ms_marker: microsaccade marker
    """
    fig_1 = plt.figure(1, figsize=(12,4))
    fig_1.clf()
    ax1 = fig_1.add_subplot(111)
    ax1.plot(t[::downsample_for_plotting], mx[::downsample_for_plotting], color='b', label='Horizontal motion')
    ax1.plot(t[::downsample_for_plotting], my[::downsample_for_plotting], color='g', label='Vertical motion')

    for i in range(len(ms_marker)):
        for j in range(2):
            if numpy.isnan(ms_marker[i][j]) == False:
                idx = int(ms_marker[i][j]-(-1)**j)
                if i==0 and j==0:
                    
                    ax1.axvline(t[idx], color='purple', label='Microsaccade', linestyle='--', alpha=0.5, linewidth=1)
                else:
                    ax1.axvline(t[idx], color='purple', linestyle='--', alpha=0.5, linewidth=1)

    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Position (arcmin)')
    ax1.legend(bbox_to_anchor=(1,1))
    ax1.set_xlim(t.min(), t.max())
    fig_1.subplots_adjust(top=0.96, left=0.07, right=0.84, bottom=0.12)
    return fig_1
        

def eye_motion_2D(t, mx, my, downsample_for_plotting=8000, cmap=plt.get_cmap('viridis')):
    '''
    Plots the motion trace in 2D space
    Parameters:
        t (ndarray): time samples.
        mx (ndarray): motion x
        my (ndarray): motion y
    '''
    import matplotlib
    from ERICA import ERICA_toolkit

    fig_2 = plt.figure(2, figsize=(6,4))
    fig_2.clf()
    ax2 = fig_2.add_subplot(111)
    
    line = ERICA_toolkit.colorline(mx[::downsample_for_plotting],my[::downsample_for_plotting], ax2,cmap=cmap, clp=255, linewidth=1)

    cbaxes = fig_2.add_axes([1.0, 0.1, 0.03, 0.8]) 
    norm = matplotlib.colors.Normalize(vmin=numpy.floor(t.min()), vmax=numpy.ceil(t.max()))
    cb1 = matplotlib.colorbar.ColorbarBase(cbaxes, cmap=cmap, norm=norm, orientation='vertical')
    cb1.set_label('Time (s)')

    ax2.set_xlabel('Horizontal position (arcmin)')
    ax2.set_ylabel('Vertical position (arcmin)')
    ax2.set_xlim(numpy.floor(mx.min())-5,numpy.ceil(mx.max())+5)
    ax2.set_ylim(numpy.floor(my.min())-5,numpy.ceil(my.max())+5)
    fig_2.subplots_adjust(top=0.96, left=0.12, right=0.94, bottom=0.12)
           
    return fig_2
        
def plot_gaussian(mu, sigma):

    import matplotlib.pyplot as plt
    import numpy as np
    import scipy.stats as stats
    import math

    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 100)
    plt.plot(x, stats.norm.pdf(x, mu, sigma))
    plt.show()