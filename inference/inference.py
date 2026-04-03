import numpy as np
import tensorflow as tf

# @tf.function(reduce_retracing=True)
def predict_step(model, batch):
    return model(batch, training=False)

#
def gaussian_kernel_3d(shape, sigma_scale=0.5):
    dz, dy, dx = shape
    z = np.linspace(-(dz//2), dz//2, dz)
    y = np.linspace(-(dy//2), dy//2, dy)
    x = np.linspace(-(dx//2), dx//2, dx)

    zz, yy, xx = np.meshgrid(z, y, x, indexing='ij')
    sigma_z = dz * sigma_scale
    sigma_y = dy * sigma_scale
    sigma_x = dx * sigma_scale

    kernel = np.exp(-(zz**2/(2*sigma_z**2) +
                      yy**2/(2*sigma_y**2) +
                      xx**2/(2*sigma_x**2)))

    return (kernel / kernel.max()).astype(np.float32)


def sliding_window_inference(volume, model,
                             patch_size=(64,96,96),
                             stride=(32,48,48),
                             batch_size=1,
                             weight=None):

    D, H, W = volume.shape
    pd, ph, pw = patch_size
    sd, sh, sw = stride

    output = np.zeros((D, H, W), dtype=np.float32)
    count_map = np.zeros((D, H, W), dtype=np.float32)

    if weight is None:
      weight = gaussian_kernel_3d(patch_size)

    # -------- FIX: cover full volume (edges included) --------
    z_range = list(range(0, max(D - pd + 1, 1), sd))
    y_range = list(range(0, max(H - ph + 1, 1), sh))
    x_range = list(range(0, max(W - pw + 1, 1), sw))

    if z_range[-1] != D - pd:
        z_range.append(D - pd)
    if y_range[-1] != H - ph:
        y_range.append(H - ph)
    if x_range[-1] != W - pw:
        x_range.append(W - pw)

    coords = [(z, y, x) for z in z_range for y in y_range for x in x_range]

    print(f"Total patches: {len(coords)}")

    # --------------------------------------------------------
    for i in range(0, len(coords), batch_size):
        batch_coords = coords[i:i+batch_size]

        patches = []
        for z, y, x in batch_coords:
            patch = volume[z:z+pd, y:y+ph, x:x+pw]
            patches.append(patch[..., np.newaxis])

        batch = np.stack(patches, axis=0)

        preds = predict_step(model, batch).numpy()

        for j, (z, y, x) in enumerate(batch_coords):
            pred = preds[j, ..., 0]

            output[z:z+pd, y:y+ph, x:x+pw] += pred * weight
            count_map[z:z+pd, y:y+ph, x:x+pw] += weight

        if i % 50 == 0:
            print(f"{i}/{len(coords)}")

    output /= (count_map + 1e-8)

    return output