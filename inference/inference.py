import numpy as np
import SimpleITK as sitk
import tensorflow as tf

@tf.function(reduce_retracing=True)
def predict_step(model, batch):
    return model(batch, training=False)

def gaussian_kernel(size=32, sigma=16):
    ax = np.linspace(-(size//2), size//2, size)
    xx, yy, zz = np.meshgrid(ax, ax, ax, indexing='ij')
    kernel = np.exp(-(xx**2 + yy**2 + zz**2)/(2*sigma**2))
    return (kernel / kernel.max()).astype(np.float32)

def sliding_window_inference(volume, model, patch_size=32, stride=16, batch_size=2, weight=None):
    D, H, W = volume.shape
    output = np.zeros((D, H, W), dtype=np.float32)
    count_map = np.zeros((D, H, W), dtype=np.float32)

    if weight is None:
        weight = gaussian_kernel(patch_size, sigma=patch_size//2)

    z_range = range(0, D - patch_size + 1, stride)
    y_range = range(0, H - patch_size + 1, stride)
    x_range = range(0, W - patch_size + 1, stride)
    coords = [(z, y, x) for z in z_range for y in y_range for x in x_range]

    total_patches = len(coords)
    print(f"Total patches to process: {total_patches}")

    for i in range(0, total_patches, batch_size):
        # --- PROGRESS TRACKER ---
        if i % 100 == 0:
            print(f"Processing patch {i}/{total_patches}...", end='\r')

        batch_coords = coords[i : i + batch_size]
        patches = [volume[z:z+patch_size, y:y+patch_size, x:x+patch_size][..., np.newaxis] for z, y, x in batch_coords]
        
        batch_data = np.stack(patches, axis=0)
        preds = predict_step(model, batch_data).numpy()

        for j, (z, y, x) in enumerate(batch_coords):
            pred = preds[j, ..., 0].astype(np.float32)
            output[z:z+patch_size, y:y+patch_size, x:x+patch_size] += pred * weight
            count_map[z:z+patch_size, y:y+patch_size, x:x+patch_size] += weight

    print("\nInference complete. Normalizing output...")
    output /= (count_map + 1e-8)
    return output