'''
Parallel within trial
'''

import os, gc, datetime, sys, yaml
from ORG import pipeline
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np
###########################

date_stamp = datetime.datetime.now()
date_stamp = date_stamp.date()
image_dir = f'./experiments/{date_stamp}'
os.makedirs(image_dir, exist_ok=True)

from ORG.utils import next_filename
_, subfolder_name = next_filename(folder = image_dir, prefix = 'image_sequence', kind = "folder")
image_dir = os.path.join(image_dir, subfolder_name)
os.makedirs(image_dir, exist_ok=True)

labels = {0: 'A', 1:'L', 2:'M', 3:'S', 4:'control'}
read_meta = False
read_config = False
if os.path.exists(os.path.join(image_dir, 'config.yaml')):
    read_config = True
if os.path.exists(os.path.join(image_dir, 'trials_meta.npy')):
    read_meta = True

SAVE_EVERY = 5

##################################
current_directory = os.getcwd()
retina = pipeline.retina(current_directory)
retina.load_config()
retina.isetbio_mosaic()
print(f'\nNumber of control trials: {retina.cone_reflectance_parameters["static_trials"]}')
print(f'Trials per condition: {retina.cone_reflectance_parameters["trials_per_condition"]}')

total_time=retina.cone_reflectance_parameters['duration_s']
frame_time = retina.myAOSLO.n_samples * retina.myAOSLO.time_resolution
n_frames = int(total_time / frame_time)
em_free = retina.cone_reflectance_parameters['em_free']
noise_free = retina.cone_reflectance_parameters['noise_free']
coherency = retina.cone_reflectance_parameters['coherency']

def check_memory():
    import psutil
    memory = psutil.virtual_memory()
    return memory.available / memory.total * 100

tasks = []
for cond in range(0,4):
    for i in range(retina.cone_reflectance_parameters['trials_per_condition']):
        tasks.append((cond, i))

for cond in range(4,5):
    for i in range(retina.cone_reflectance_parameters['static_trials']):
        tasks.append((cond, i))

def frame_task(trial_idx, frnum, motion_framewise, reflectance_framewise, retina_window_template, noise_free):
    retina_window = retina_window_template.copy()
    retina_window[5] = reflectance_framewise
    if noise_free is False:
        image = retina.make_aoslo_image(retina_window, motion_framewise)
    elif noise_free is True:
        image = retina.make_aoslo_image(retina_window, motion_framewise, noise=False)
    return trial_idx, frnum, image

if read_meta == False:
    trials_meta = {
        "trial_idx": [],
        "cond": [],
        "trial_num": [],
        "motions": [],
        "reflectance": [],
    }

    counter = 0
    for trial_idx, (cond, trial_num) in enumerate(tasks):
        sys.stdout.write('Preparing trial meta %03i of %03i\r'%(counter+1, len(tasks)))
        sys.stdout.flush()
        motions, reflectance = retina.prepare_trial_params(
            cond,
            total_time=retina.cone_reflectance_parameters['duration_s'],
            em_free=em_free,
            coherency=coherency
        )
        trials_meta["trial_idx"].append(trial_idx)
        trials_meta["cond"].append(cond)
        trials_meta["trial_num"].append(trial_num)
        trials_meta["motions"].append(motions)
        trials_meta["reflectance"].append(reflectance)
        counter+=1

        if counter % SAVE_EVERY == 0:
            np.save(
                os.path.join(image_dir, "trials_meta_partial.npy"),
                trials_meta,
                allow_pickle=True
            )


    os.replace(os.path.join(image_dir, "trials_meta_partial.npy"),
            os.path.join(image_dir, "trials_meta.npy"))
    
    print(f'\nTrial meta saved to {os.path.join(image_dir, "trials_meta.npy")}.')

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
    }
    with open(os.path.join(image_dir, 'config.yaml'), 'w') as file:
        yaml.safe_dump(config_dict, file, sort_keys=False)

    print(f'Simulation config file saved to {os.path.join(image_dir, "config.yaml")}')

elif read_meta == True:
    # read config and reset ERICA

    # read pre-saved trial metas to avoid calculating it again
    print(f'Reading pre-calculated trials meta: {os.path.join(image_dir, "trials_meta.npy")}')
    trials_meta = np.load(os.path.join(image_dir, "trials_meta.npy"), allow_pickle=True).item()


if read_config == True:
    print(f'\nResetting ERICA with pre-saved config: {os.path.join(image_dir, "config.yaml")}')
    with open(os.path.join(image_dir, 'config.yaml'), 'r') as f:
        config = yaml.safe_load(f)

    retina.cone_reflectance_parameters = config["reflectance_parameters"]
    system_parameters = config["system_parameters"]
    retina.myEyeMotion.parameters = config["eye_motion_parameters"]

    from ERICA import AOSLO_system
    system_parameters_file = 'system_parameters_myAOSLO.yaml'
    aoslo = AOSLO_system.AOSLO(os.path.normpath(os.path.join(retina.current_directory, 'System_parameters', system_parameters_file)))
    aoslo.parameters = system_parameters
    aoslo.setScanners()
    retina.myAOSLO = aoslo

print('\nReady')

example_img = retina.make_aoslo_image()
images = np.zeros((n_frames, example_img.shape[0], example_img.shape[1]), dtype=example_img.dtype)
retina_window_template = retina.retina_window_isetbio.copy()

H, W = example_img.shape
dtype = example_img.dtype

memmaps = {}
frames_remaining = {}
for trial_idx, cond, trial_num, motions in zip(trials_meta['trial_idx'], 
                                            trials_meta['cond'],
                                            trials_meta['trial_num'],
                                            trials_meta['motions']
                                            ):
    fname = os.path.join(image_dir, f"{labels[cond]}_{trial_num+1:03d}.npy")
    mem = np.lib.format.open_memmap(fname, mode='w+', dtype=dtype, shape=(n_frames, H, W))
    mem.flush()
    memmaps[trial_idx] = (fname, mem)
    frames_remaining[trial_idx] = n_frames

print('\nGenerating image sequences...')
available_memory = check_memory()
suggested_workers = max(1, min(os.cpu_count(), int(available_memory/10)))
max_workers = min(suggested_workers, 6)
print(f'Suggested workers: {suggested_workers}, Maximum workers used: {max_workers}')

futures = []
with ProcessPoolExecutor(max_workers=max_workers) as ex:
    # for trial_idx, cond, trial_num, motions, reflectance in trials_meta:
    for trial_idx, cond, trial_num, motions, reflectance in zip(trials_meta['trial_idx'], 
                                                                trials_meta['cond'],
                                                                trials_meta['trial_num'],
                                                                trials_meta['motions'],
                                                                trials_meta['reflectance']):
        for fr in range(n_frames):
            futures.append(
                ex.submit(
                    frame_task,
                    trial_idx, 
                    fr,
                    motions[fr],
                    reflectance[:, fr],
                    retina_window_template,
                    noise_free
                )
            )

    with tqdm(total=len(futures), desc="Generating frames") as pbar:
        write_counts = {tid: 0 for tid in memmaps.keys()}
        try:
            for fut in as_completed(futures):
                t_idx, f_idx, img = fut.result()
                fname, mem = memmaps[t_idx]
                # write into memmap (main process only)
                mem[f_idx] = img
                write_counts[trial_idx] += 1
                mem.flush()   # ensure written to disk
                pbar.update(1)

                # if this trial completed, finalize 
                if frames_remaining[trial_idx] == 0:
                    mem.flush()
                    del mem
                    gc.collect()  

        except KeyboardInterrupt:
            # Attempt to flush all partial memmaps before exiting
            print("\nKeyboardInterrupt: flushing partial memmaps...")
            for tid, (final_path, mem) in list(memmaps.items()):
                if mem is not None:
                    try:
                        mem.flush()
                    except Exception:
                        pass
            raise
    
    del retina
    del trials_meta
    gc.collect()
        
