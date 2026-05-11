'''
Parallel within trial
'''

import os, time
from ORG import pipeline
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np
###########################

video = False
condition = 0

##################################
current_directory = os.getcwd()
retina = pipeline.retina(current_directory)
retina.load_config()
retina.isetbio_mosaic()

def check_memory():
    import psutil
    memory = psutil.virtual_memory()
    return memory.available / memory.total * 100

available_memory = check_memory()
suggested_workers = max(1, min(os.cpu_count(), int(available_memory/10)))
max_workers = min(suggested_workers, 6)
print(f'Suggested workers: {suggested_workers}, Maximum workers used: {max_workers}')

motions, reflectance = retina.prepare_trial_params(condition, total_time=retina.cone_reflectance_parameters['duration_s'], em_free=retina.cone_reflectance_parameters['em_free'], coherency=retina.cone_reflectance_parameters['coherency'])

def frame_task(frnum, motion_framewise, reflectance_framewise, retina_window_template):
    retina_window = retina_window_template.copy()
    retina_window[5] = reflectance_framewise
    image = retina.make_aoslo_image(retina_window, motion_framewise)
    return frnum, image

n_frames = len(motions)
example_img = retina.make_aoslo_image()
images = np.zeros((n_frames, example_img.shape[0], example_img.shape[1]), dtype=example_img.dtype)
retina_window_template = retina.retina_window_isetbio.copy()

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


data_dir = './trials'
os.makedirs(data_dir, exist_ok=True)
labels = {0: 'A', 1:'L', 2:'M', 3:'S', 4:'C'}

# prefix = f"{retina.cone_reflectance_parameters['exp_type']}_{labels[condition]}_"
prefix = f"{labels[condition]}"
suffix = '.npy'

from ORG.utils import next_filename
fnamef, fname = next_filename(data_dir, prefix, suffix)

# np.save(os.path.join(data_dir, fname+'_motions.npy'), np.asarray(motions))
# print(f"Motions saved to file: {os.path.join(data_dir, fname+'_em_trace.npy')}")

np.save(os.path.join(data_dir, fnamef), np.asarray(images))
print(f'Save to file: {os.path.join(data_dir, fnamef)}')

if video == True:
    import imagesToMovie
    imagesToMovie.createVideo(np.asarray(images), fname, path=data_dir, ext='.avi',nm=1)
    print(f"Save to video name: {os.path.join('./videos', fname)}.avi")
