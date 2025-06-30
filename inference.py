import numpy as np
from model import unet_3d
from utils import vessel_density, dice_coef
from dataset import load_nifti, apply_clahe_3d, resize_volume

from tensorflow.keras.models import load_model

def run_inference(model_path, image_path, shape=(128,128,64)):
    model = load_model(model_path, compile=False)
    img = load_nifti(image_path)
    img = apply_clahe_3d(img)
    img = resize_volume(img, shape)
    img = img[np.newaxis, ..., np.newaxis]
    pred = model.predict(img)[0]
    mask = (pred > 0.5).astype(np.uint8)
    density = vessel_density(mask)
    return mask, density

# Example usage:
# mask, density = run_inference('../model_3d_unet.h5', '../data/imagesTr/some_image.nii.gz')
# print('Vessel density:', density)
