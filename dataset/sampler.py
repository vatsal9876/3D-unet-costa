import numpy as np

# sampler file

# sampler file

def extract_patch(img, msk, patch_size=96, min_vessel_voxels=200,bg_threshold=20, vessel_prob=0.7, max_tries=10 , depth_patch_size=64):

    D, H, W = img.shape

    if isinstance(patch_size, int):
        ph = pw = patch_size
        pd=depth_patch_size
    else:
        pd, ph, pw = patch_size

    msk = (msk > 0).astype(np.uint8)
    vessel_coords = np.argwhere(msk > 0)

    if len(vessel_coords) > 0 and np.random.rand() < vessel_prob:
        # vessel-centered
        for _ in range(max_tries):
            z, y, x = vessel_coords[np.random.randint(len(vessel_coords))]

            d = np.clip(z - pd//2, 0, max(0, D - pd))
            h = np.clip(y - ph//2, 0, max(0, H - ph))
            w = np.clip(x - pw//2, 0, max(0, W - pw))

            ip = img[d:d+pd, h:h+ph, w:w+pw]
            mp = msk[d:d+pd, h:h+ph, w:w+pw]

            if np.sum(mp) >= min_vessel_voxels:
                return ip, mp

    else:
        # random
        for _ in range(max_tries):
            d = np.random.randint(0, max(1, D - pd + 1))
            h = np.random.randint(0, max(1, H - ph + 1))
            w = np.random.randint(0, max(1, W - pw + 1))

            ip = img[d:d+pd, h:h+ph, w:w+pw]
            mp = msk[d:d+pd, h:h+ph, w:w+pw]

            if np.sum(mp) <= bg_threshold:
                return ip, mp

    return ip, mp
