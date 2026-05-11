
import subprocess

num_blocks = 3
for i in range(num_blocks):
    print(f"Starting block {i+1:02d} / {num_blocks:02d}")
    subprocess.run(["python", "run_trials_parallel_wrapped.py"])