import numpy as np

def extract_patch(img, msk, patch_size=48, min_vessel_voxels=50, vessel_prob=0.5, max_tries=2):

    D, H, W = img.shape

    if isinstance(patch_size, int):
        pd = ph = pw = patch_size
    else:
        pd, ph, pw = patch_size

    msk = (msk > 0).astype(np.uint8)
    vessel_coords = np.argwhere(msk > 0)

    def get_patch():
        if len(vessel_coords) > 0 and np.random.rand() < vessel_prob:
            # vessel-centered
            z, y, x = vessel_coords[np.random.randint(len(vessel_coords))]

            d = np.clip(z - pd//2, 0, max(0, D - pd))
            h = np.clip(y - ph//2, 0, max(0, H - ph))
            w = np.clip(x - pw//2, 0, max(0, W - pw))
        else:
            # random
            d = np.random.randint(0, max(1, D - pd + 1))
            h = np.random.randint(0, max(1, H - ph + 1))
            w = np.random.randint(0, max(1, W - pw + 1))

        ip = img[d:d+pd, h:h+ph, w:w+pw]
        mp = msk[d:d+pd, h:h+ph, w:w+pw]

        return ip, mp

    # -----------------------------
    # generate → validate → retry
    # -----------------------------
    for _ in range(max_tries):
        ip, mp = get_patch()

        if np.sum(mp) >= min_vessel_voxels:
            return ip, mp

    # fallback (last generated patch)
    return ip, mp