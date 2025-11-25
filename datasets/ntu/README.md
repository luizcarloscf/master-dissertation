# NTU RGB+D Datasets

This section explains how to download and preprocess both **NTU RGB+D 60** and **NTU RGB+D 120**. The goal is to create clean, normalized skeleton sequences suitable for training action recognition models.

## Download Datasets

1. Request access to the datasets at:  
   https://rose1.ntu.edu.sg/dataset/actionRecognition

2. After receiving access, download the **skeleton-only** versions of the datasets:

   - `nturgbd_skeletons_s001_to_s017.zip` → **NTU RGB+D 60**
   - `nturgbd_skeletons_s018_to_s032.zip` → **NTU RGB+D 120**

3. Extract both archives into the directory: `./ntu-raw`

## Directory Structure

Your dataset directory should follow the structure below:

```bash
- datasets/
  - ntu/
    - ntu-60/
    - ntu-120/
    - ntu-raw/
         - nturgb+d_skeletons/     # from `nturgbd_skeletons_s001_to_s017.zip`
            - S001C001P001R001A001.skeleton
            ...
        - nturgb+d_skeletons120/  # from `nturgbd_skeletons_s018_to_s032.zip`
            - S018C001P001R001A001.skeleton
            ... 
```

- `ntu-60/` and `ntu-120/` will contain preprocessed outputs.
- `nturgbd_raw/` stores the raw skeleton sequences provided by NTU.

---

## Preprocessing NTU RGB+D

To generate the fully processed versions of NTU RGB+D 60 or NTU RGB+D 120:

```bash
cd ntu-60/   # or cd ntu-120/
```

Then run the preprocessing pipeline:

1. Extract skeleton data for all performers
```bash
python3 get_raw_skes_data.py
```

2. Remove corrupted or invalid skeleton sequences
```bash
python3 get_raw_denoised_data.py
```

3. Normalize sequences by transforming skeletons to the center of the first frame
```bash
python3 seq_transformation.py
```

What This Pipeline Does:
- Collects all raw .skeleton files and converts them to a unified numpy-like format
- Automatically filters corrupted videos (missing joints, incomplete frames, wrong tracking, etc.)
- Aligns each skeleton to a consistent origin (root-centered coordinates)
- Produces clean, standardized sequences ideal for training AI models

## Acknowledgement

The preprocessing scripts used here are partially based on the excellent work from [MAMP](https://github.com/maoyunyao/MAMP).