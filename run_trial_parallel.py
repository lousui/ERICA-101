'''
Parallel within trial.
Sirui Liu, Jan 2026
'''

import os
from ORG import pipeline
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np
import yaml

# Set parameters 
video = True
fname = 'with_em_noise_free.npz'
data_dir = './trials'
os.makedirs(data_dir, exist_ok=True)

# Load config 
current_directory = os.getcwd()
retina = pipeline.retina(current_directory)
retina.load_config()
retina.isetbio_mosaic()

# check resources
def check_memory():
    import psutil
    memory = psutil.virtual_memory()
    return memory.available / memory.total * 100

available_memory = check_memory()
suggested_workers = max(1, min(os.cpu_count(), int(available_memory/10)))
max_workers = min(suggested_workers, 6)
print(f'Suggested workers: {suggested_workers}, Maximum workers used: {max_workers}')

# Get per frame eye motion and cone reflectance 
motions, reflectance = \
    retina.prepare_trial_params(
        condition=4,    # always static trial
        total_time=retina.cone_reflectance_parameters['duration_s'], 
        em_free=retina.cone_reflectance_parameters['em_free'], 
        coherency=retina.cone_reflectance_parameters['coherency'],
        )

def frame_task(frnum, motion_framewise, reflectance_framewise, retina_window_template):
    retina_window = retina_window_template.copy()
    retina_window[5] = reflectance_framewise
    image = retina.make_aoslo_image(retina_window, motion_framewise)
    return frnum, image

# pre-allocation
n_frames = len(motions)
example_img, cones_scanned, groundtruth_centroids = retina.make_aoslo_image(return_cones=True)
images = np.zeros((n_frames, example_img.shape[0], example_img.shape[1]), dtype=example_img.dtype)
retina_window_template = retina.retina_window_isetbio.copy()
cone_types = retina.mosaic['cone_types'][cones_scanned]
reflectance = np.asarray(reflectance)
reflectance = reflectance[cones_scanned, :]

with ProcessPoolExecutor(max_workers=max_workers) as executor:
    futures = []
    for frnum in range(n_frames):
        reflectance_framewise = reflectance[:, frnum]
        motion_framewise = motions[frnum]
        future = executor.submit(frame_task, frnum, motion_framewise, reflectance_framewise, retina_window_template)
        futures.append(future)
    
    with tqdm(total=n_frames, desc='Generating frames') as pbar:
        for future in as_completed(futures):
            frnum, img = future.result()
            images[frnum] = img
            pbar.update(1)

# save image sequences and other variables of interest
fpath = os.path.join(data_dir, fname)
np.savez(fpath,
         frames = np.asarray(images),
         motions = np.asarray(motions)
         )
print(f'Save to file: {fpath}')

# dump settings to one config
from ORG.utils import to_python
config_dict = {
    "reflectance_parameters": to_python(retina.cone_reflectance_parameters),
    "system_parameters": to_python(retina.myAOSLO.parameters),
    "eye_motion_parameters": to_python(retina.myEyeMotion.parameters),
}
with open(os.path.join(data_dir, fname.replace('.npz', '.yaml')), 'w') as file:
    yaml.safe_dump(config_dict, file, sort_keys=False)

# write to video
if video == True:
    from ORG.utils import write_grayscale_avi
    video_path =  os.path.join('./videos', fname.replace('.npz', '.mp4'))
    write_grayscale_avi(np.asarray(images), video_path, fps=30)
    print(f"Save to video name: {video_path}")
