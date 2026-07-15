'''
Parallel within trial
'''

import os
from ORG import pipeline
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np
import yaml
###########################

video = True
condition = 4   # control
clip = False

# fname = 'noise_but_em_free.npz'
fname = 'with_em.npz'

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

motions, reflectance = \
    retina.prepare_trial_params(
        condition, 
        total_time=retina.cone_reflectance_parameters['duration_s'], 
        em_free=retina.cone_reflectance_parameters['em_free'], 
        coherency=retina.cone_reflectance_parameters['coherency'],
        drift_only=retina.cone_reflectance_parameters['drift_only'],
        )

def frame_task(frnum, motion_framewise, reflectance_framewise, retina_window_template):
    retina_window = retina_window_template.copy()
    retina_window[5] = reflectance_framewise
    image = retina.make_aoslo_image(retina_window, motion_framewise, clip=clip)
    return frnum, image

n_frames = len(motions)
example_img, cones_scanned, groundtruth_centroids = retina.make_aoslo_image(return_cones=True, clip=clip)
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
suffix = '.npz'

# from ORG.utils import next_filename
# fnamef, fname = next_filename(data_dir, prefix, suffix)

# np.save(os.path.join(data_dir, fname+'_motions.npy'), np.asarray(motions))
# print(f"Motions saved to file: {os.path.join(data_dir, fname+'_em_trace.npy')}")

cone_types = retina.mosaic['cone_types'][cones_scanned]
reflectance = np.asarray(reflectance)
reflectance = reflectance[cones_scanned, :]


fpath = os.path.join(data_dir, fname)
# np.save(os.path.join(data_dir, fnamef), np.asarray(images))
np.savez(fpath,
         frames = np.asarray(images),
         ref = np.asarray(example_img),
         cone_types = np.asarray(cone_types),
         groundtruth_centroids = np.asarray(groundtruth_centroids),
         reflectance = np.asarray(reflectance),
         )
print(f'Save to file: {fpath}')

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

config_dict = {
    "reflectance_parameters": to_python(retina.cone_reflectance_parameters),
    "system_parameters": to_python(retina.myAOSLO.parameters),
    "eye_motion_parameters": to_python(retina.myEyeMotion.parameters),
    "clip": clip,
}
with open(os.path.join(data_dir, fname.replace('.npz', '.yaml')), 'w') as file:
    yaml.safe_dump(config_dict, file, sort_keys=False)

if video == True:
    import cv2
    def write_grayscale_avi(arr: np.ndarray, path: str, fps: int = 30):
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
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # good default for AVI
        writer = cv2.VideoWriter(path, fourcc, fps, (w, h), isColor=False)

        if not writer.isOpened():
            raise RuntimeError("Could not open video writer")

        for frame in arr:
            writer.write(frame)  # single-channel grayscale frame

        writer.release()


    video_path =  os.path.join('./videos', fname.replace('.npz', '.mp4'))
    write_grayscale_avi(np.asarray(images), video_path, fps=30)
    print(f"Save to video name: {video_path}")
