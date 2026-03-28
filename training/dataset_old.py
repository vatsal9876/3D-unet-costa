import os
import sys
import numpy as np
import cv2
import SimpleITK as sitk

# Allow imports from project root when running scripts from `training/`.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from patches import balanced_patch_sampling


# -------------------------------------------------
# Load MHD Volume
# -------------------------------------------------
def load_mhd(path):
    img = sitk.ReadImage(path)
    volume = sitk.GetArrayFromImage(img)   # shape: (Z, Y, X)
    volume = volume.astype(np.float32)
    return volume


# -------------------------------------------------
# CLAHE on each slice
# -------------------------------------------------
def apply_clahe_3d(volume):

    volume_clahe = np.zeros_like(volume)

    for i in range(volume.shape[0]):  # iterate over Z slices
        slice_ = volume[i, :, :]

        slice_norm = cv2.normalize(
            slice_, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        volume_clahe[i, :, :] = clahe.apply(slice_norm)

    return volume_clahe


# -------------------------------------------------
# Normalize to [0,1]
# -------------------------------------------------
def normalize(volume):

    min_val = np.min(volume)
    max_val = np.max(volume)

    volume = (volume - min_val) / (max_val - min_val + 1e-8)

    return volume


# -------------------------------------------------
# Main Dataset Loader
# -------------------------------------------------
def load_dataset(images_dir, labels_dir):

    image_patches_train = []
    image_patches_test = []

    mask_patches_train = []
    mask_patches_test = []

    images_file = sorted(
        [f for f in os.listdir(images_dir) if f.endswith(".mhd")]
    )

    # Random 20% scans for testing
    test_list = np.random.choice(
        range(len(images_file)),
        size=len(images_file) // 5,
        replace=False
    )

    for idx, file in enumerate(images_file):

        image_path = os.path.join(images_dir, file)
        mask_path = os.path.join(labels_dir, file)

        if not os.path.exists(mask_path):
            print(f"Mask missing for {file}")
            continue

        # Load volumes
        image = load_mhd(image_path)
        mask = load_mhd(mask_path)

        # Preprocessing
        image = apply_clahe_3d(image)
        image = normalize(image)

        mask = (mask > 0).astype(np.float32)

        # Patch sampling
        img_p, mask_p = balanced_patch_sampling(
            image,
            mask,
            patch_size=64,
            num_patches=40
        )

        # Train / Test split
        if idx in test_list:
            image_patches_test.append(img_p)
            mask_patches_test.append(mask_p)
        else:
            image_patches_train.append(img_p)
            mask_patches_train.append(mask_p)

        print(f"Processed {file} → patches: {img_p.shape}")

    # -------------------------------------------------
    # Concatenate patches
    # -------------------------------------------------
    X_train = np.concatenate(image_patches_train, axis=0)
    Y_train = np.concatenate(mask_patches_train, axis=0)

    X_test = np.concatenate(image_patches_test, axis=0)
    Y_test = np.concatenate(mask_patches_test, axis=0)

    # -------------------------------------------------
    # Add channel dimension for Conv3D
    # -------------------------------------------------
    X_train = X_train[..., np.newaxis]
    Y_train = Y_train[..., np.newaxis]

    X_test = X_test[..., np.newaxis]
    Y_test = Y_test[..., np.newaxis]

    # -------------------------------------------------
    # Dataset summary
    # -------------------------------------------------
    print("\nDataset Summary")
    print("Train images:", X_train.shape)
    print("Train masks :", Y_train.shape)

    print("Test images :", X_test.shape)
    print("Test masks  :", Y_test.shape)

    return X_train, Y_train, X_test, Y_test