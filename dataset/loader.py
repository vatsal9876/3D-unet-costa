import os
import numpy as np
import nibabel as nib


# ---------- LOAD NIFTI ----------
# loader file

def load_nifti(path):
    nii = nib.load(path)
    return nii.get_fdata().astype(np.float32)


# ---------- PREPROCESS ----------

def preprocess(volume, mask):
    # (H, W, D) → (D, H, W)
    volume = np.transpose(volume, (2, 0, 1))
    mask   = np.transpose(mask, (2, 0, 1))

    # normalize volume
    volume = (volume - np.mean(volume)) / (np.std(volume) + 1e-8)

    # binarize mask
    mask = (mask > 0).astype(np.float32)

    return volume, mask



def load_volume(volume_path, mask_path):
    volume = load_nifti(volume_path)
    mask   = load_nifti(mask_path)

    volume, mask = preprocess(volume, mask)
    
    return volume, mask 