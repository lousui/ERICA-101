# ERICA-101 

This repository is a stripped-down, streamlined version of a modified **ERICA**, specifically designed to generate ERICA image sequences using the custom **ORG module**. 

While the original [ERICA](https://github.com/LauraKateYoung/ERICA) provides a detailed tutorial, this repository focuses on the ORG module—a wrapper developed for experiments requiring **realistic cone reflectance changes** and **continuous video output** instead of single, static images.

---

## Setup & Environment

This repository is fully configured with **`uv`**. 

### 1. Install uv
Follow the instructions on the official `uv` installation page. For macOS users, the simplest way is via Homebrew:
```bash
brew install uv
```

### 2. Setup the Environment
Clone the repository, navigate into the directory, and run the following command to automatically create the virtual environment and install all dependencies:
```bash
git clone https://github.com/lousui/ERICA-101.git
cd ERICA-101
uv sync
```

---

## Getting Started

The easiest way to understand the pipeline is through the interactive walkthrough:
```bash
uv run jupyter lab HelloErica.ipynb
```
Open **`HelloErica.ipynb`** to see a quick demonstration of how to configure and generate your first image sequence.

---

## Key Features & Workflow

### 1. Advanced Mosaic Generation (`ISETBio`)
* **The Limitation:** The original ERICA uses pre-generated retina mosaics via a reaction-diffusion model, which are limited in size.
* **Our Solution:** This pipeline offers an alternative approach using **ISETBio** (a simulation tool containing rich metadata about the cone mosaic, including cone types, outer segment length, etc.). 
* **Data Provided:** A pre-computed **15° x 15° ISETBio mosaic centered at the fovea** is available at `../isetbio_retina/mosaic_00.mat`.

### 2. Realistic Cone Intensities (Interference vs. Normal)
Unlike the original ERICA which randomly samples cone intensities from a normal distribution, this pipeline allows you to toggle between two modes:
* `normal`: Cone intensities sampled from a standard normal distribution.
* `interference`: Realistic cone intensities calculated based on their actual **outer segment length**.

### 3. Parallel Batch Generation & Video Export
Generating a single image takes a few seconds. For large-scale data generation, a parallelized script is provided to speed up the process:
```bash
uv run python run_trial_parallel.py
```
Note that this script requires modification in save file name and folder. 

**Features of `run_trial_parallel.py`:**
* Generates a long trial quickly using parallel processing.
* Saves image sequences and eye motions into a compact `.npz` file.
* Dumps all configuration settings into a single `.yaml` file.
* Offers an optional feature to directly write out the sequence as an `.mp4` video.
