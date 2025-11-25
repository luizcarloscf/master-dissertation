# Human-Robot Interaction for Intelligent Spaces (HRI-IS) Dataset

The **HRI-IS Dataset** is a proprietary dataset developed for research in human–robot interaction within Intelligent Spaces. Due to privacy and usage restrictions, access is controlled and requires formal authorization.

## How to Request Access

To obtain a copy of the dataset:

1. Send an email to **[labvisio.ufes@gmail.com](mailto:labvisio.ufes@gmail.com)** requesting access to the **HRI-IS Dataset**.
2. A member of the LabVISIO research group will reply with a **“Terms and Conditions”** document.
3. Carefully read, sign, and return the document.
4. After verification, you will receive a **temporary download link** containing the dataset files.

## Important Legal Notice

Access to the dataset is granted **exclusively for academic and non-commercial research purposes**.  
By requesting access, you agree to the following restrictions:

- **Redistribution of the dataset is strictly prohibited.**  
  Sharing, uploading, or transferring the data to third parties without explicit written permission is illegal.

- **Images and videos from the dataset may NOT be published** in any form, including:  
  - research papers  
  - theses or dissertations  
  - supplementary videos  
  - conference presentations  
  - websites, social media, or any online platform  

Any form of public disclosure of the raw imagery is considered a violation of the Terms and Conditions. Failure to comply with these rules may result in legal consequences and the immediate revocation of dataset privileges.

## Extracting the Dataset

Once you receive and download the file `hir-is.zip`, simply extract its contents into this directory. It should contain the following files for each person:

```
- hri-is
    - p001g01c00.mp4
    - p001g01c01.mp4
    - p001g01c02.mp4
    - p001g01c03.mp4
    - p001g01_timestamps.json # timestamps of captured images
    - p001g01_spots.json # annotations, with gestures begin and end
    - coco17
      - p001g01_3d.npy # numpy vector with filtered 3D skeleton
      - ...
    - coco33
      - p001g01_3d.npy # numpy vector with filtered 3D skeleton
      - ...
```



