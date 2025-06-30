import os
import numpy as np
import nibabel as nib
from skimage.transform import resize

def load_nifti(path):
    return nib.load(path).get_fdata()

def apply_clahe_3d(volume):
    import cv2
    volume_clahe = np.zeros_like(volume)
    for i in range(volume.shape[2]):
        slice_ = volume[:, :, i]
        slice_norm = cv2.normalize(slice_, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        volume_clahe[:, :, i] = clahe.apply(slice_norm)
    return volume_clahe

def resize_volume(vol, shape=(128,128,64)):
    return resize(vol, shape, preserve_range=True, anti_aliasing=True)

def load_dataset(images_dir, labels_dir, max_samples=None, shape=(128,128,64)):
    images, masks = [], []
    files = sorted([f for f in os.listdir(images_dir) if f.endswith('.nii.gz')])
    if max_samples:
        files = files[:max_samples]
    for file in files:
        img = load_nifti(os.path.join(images_dir, file))
        img = apply_clahe_3d(img)
        img = resize_volume(img, shape)
        images.append(img[..., np.newaxis])
        label_file = file.replace('_0000.nii.gz', '.nii.gz')
        mask = load_nifti(os.path.join(labels_dir, label_file))
        mask = resize_volume(mask, shape)
        masks.append(mask[..., np.newaxis])
    return np.array(images), np.array(masks)
