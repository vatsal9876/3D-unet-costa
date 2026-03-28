import numpy as np
from .loader import load_volume, normalize_ct
from .sampler import extract_patch
import gc

def data_generator(image_paths, mask_paths, patch_size=48, min_vessel_voxels=50, batch_size=1):
    n = len(image_paths)

    while True:
        imgs = []
        msks = []

        for _ in range(batch_size):
            j = np.random.randint(0, n)

            img, msk = load_volume(image_paths[j], mask_paths[j])

            ip, mp = extract_patch(
                img, msk,
                patch_size=patch_size,
                min_vessel_voxels=min_vessel_voxels
            )

            # ---- FIXES ----

            ip = ip.astype(np.float32)
            mp = mp.astype(np.float32)

            # normalize mask if needed
            if mp.max() > 1:
                mp = mp / 255.0

            # FORCE channel dim (no conditions)
            ip = ip[..., np.newaxis]
            mp = mp[..., np.newaxis]

            # strict shape check
            assert ip.shape == mp.shape, f"{ip.shape} vs {mp.shape}"

            imgs.append(ip)
            msks.append(mp)

        imgs = np.stack(imgs, axis=0)
        msks = np.stack(msks, axis=0)

        yield imgs, msks
        del imgs, msks, ip, mp, img, msk
        gc.collect()