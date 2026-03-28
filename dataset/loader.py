import numpy as np
import cv2
import SimpleITK as sitk

def load_mhd(path):
    img = sitk.ReadImage(path)
    return sitk.GetArrayFromImage(img).astype(np.float32)

# def apply_clahe_3d(volume):

#     out = np.zeros_like(volume)

#     for i in range(volume.shape[0]):

#         sl = volume[i]

#         sl = cv2.normalize(sl, None, 0,255,cv2.NORM_MINMAX).astype(np.uint8)

#         clahe = cv2.createCLAHE(2.0,(8,8))

#         out[i] = clahe.apply(sl)

#     return out

# def normalize(volume):

#     vmin = volume.min()
#     vmax = volume.max()

#     return (volume-vmin)/(vmax-vmin+1e-8)

def normalize_ct(volume):
    volume = np.clip(volume, -1000, 400)
    volume = (volume + 1000) / 1400
    return volume.astype(np.float32)


def load_volume(img_path, mask_path):

    img = load_mhd(img_path)
    mask = load_mhd(mask_path)

    img= normalize_ct(img)
    mask = (mask>0).astype(np.float32)

    return img, mask


# from functools import lru_cache

# # @lru_cache(maxsize=3)
# def load_cached_volume(img_path, mask_path):
#     img, msk = load_volume(img_path, mask_path)
#     img = normalize_ct(img)
#     return img, msk