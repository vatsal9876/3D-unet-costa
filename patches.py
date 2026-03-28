import numpy as np

def extract_patch(volume, center, patch_size):
    z, y, x = center
    half = patch_size // 2

    z1, z2 = z - half, z + half
    y1, y2 = y - half, y + half
    x1, x2 = x - half, x + half

    # Boundary safety
    if z1 < 0 or y1 < 0 or x1 < 0:
        return None
    if z2 > volume.shape[0] or y2 > volume.shape[1] or x2 > volume.shape[2]:
        return None

    return volume[z1:z2, y1:y2, x1:x2]

def balanced_patch_sampling(volume, mask, patch_size=64, num_patches=100):
    img_patches = []
    mask_patches = []
    half = patch_size // 2

    if volume.ndim != 3 or mask.ndim != 3:
        raise ValueError("balanced_patch_sampling expects 3D arrays (D, H, W).")

    D, H, W = volume.shape
    if D < patch_size or H < patch_size or W < patch_size:
        empty = np.empty((0, patch_size, patch_size, patch_size), dtype=np.float32)
        return empty, empty

    # Find vessel voxels
    vessel_coords = np.argwhere(mask > 0)

    if len(vessel_coords) == 0:
        empty = np.empty((0, patch_size, patch_size, patch_size), dtype=np.float32)
        return empty, empty

    # ---- 50% vessel-centered patches ----
    num_vessel = num_patches // 2

    for _ in range(num_vessel):
        center = vessel_coords[np.random.randint(len(vessel_coords))]
        img_patch = extract_patch(volume, center, patch_size)
        mask_patch = extract_patch(mask, center, patch_size)

        if img_patch is not None:
            img_patches.append(img_patch)
            mask_patches.append(mask_patch)

    # ---- 50% random background patches ----
    num_background = num_patches // 2

    for _ in range(num_background):
        center = [
            np.random.randint(half, D - half + 1),
            np.random.randint(half, H - half + 1),
            np.random.randint(half, W - half + 1)
        ]

        img_patch = extract_patch(volume, center, patch_size)
        mask_patch = extract_patch(mask, center, patch_size)

        if img_patch is not None:
            img_patches.append(img_patch)
            mask_patches.append(mask_patch)

    img_patches = np.array(img_patches, dtype=np.float32)
    mask_patches = np.array(mask_patches, dtype=np.float32)

    return img_patches, mask_patches

def build_patches(X,Y):
    image_patches, mask_patches = [], []
    for img, mask in zip(X, Y):
        if img.ndim == 4 and img.shape[-1] == 1:
            img = np.squeeze(img, axis=-1)
        if mask.ndim == 4 and mask.shape[-1] == 1:
            mask = np.squeeze(mask, axis=-1)

        img_p , mask_p= balanced_patch_sampling(img,mask,patch_size=64,num_patches=100)
        if len(img_p) == 0:
            continue

        image_patches.append(img_p)
        mask_patches.append(mask_p)

    if not image_patches:
        raise ValueError("No patches were generated. Check mask content and patch size.")

    X=np.concatenate(image_patches,axis=0)
    Y=np.concatenate(mask_patches,axis=0)

    X=X[...,np.newaxis]
    Y=Y[...,np.newaxis]
    return X,Y
