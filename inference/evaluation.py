import numpy as np
import tensorflow as tf
import SimpleITK as sitk
import matplotlib.pyplot as plt
import os
import sys
from skimage import morphology

# GPU Config for 4GB VRAM
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            print(gpu, "configured for memory growth.")
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy('mixed_float16')
    except RuntimeError as e:
        print(e)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from inference import sliding_window_inference, gaussian_kernel
from dataset.loader import load_volume 
from utils.losses import combined_loss

def load_trained_model(model_path):
    return tf.keras.models.load_model(model_path, compile=False)

def compute_metrics(pred, gt):
    pred, gt = pred.astype(bool), gt.astype(bool)
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    return dice, precision, recall

def visualize_slice(volume, pred, gt, save_path=None):
    slice_idx = volume.shape[0] // 2
    plt.figure(figsize=(15,5))
    plt.subplot(1,3,1); plt.imshow(volume[slice_idx], cmap='gray'); plt.title("CT")
    plt.subplot(1,3,2); plt.imshow(pred[slice_idx], cmap='gray'); plt.title("Pred")
    plt.subplot(1,3,3); plt.imshow(gt[slice_idx], cmap='gray'); plt.title("GT")
    if save_path: plt.savefig(save_path)
    else: plt.show()
    plt.close('all')

def evaluate_model(model, test_img_paths, test_mask_paths, weight, save_vis=False):
    dice_scores, precision_scores, recall_scores = [], [], []

    for idx, (img_path, mask_path) in enumerate(zip(test_img_paths, test_mask_paths)):
        file_name = os.path.basename(img_path)
        print(f"\n--- STARTING EVALUATION: {file_name} ---")
        
        volume, gt = load_volume(img_path, mask_path)
        gt_bin = (gt > 0).astype(np.uint8)

        # Run Inference
        pred = sliding_window_inference(volume, model, patch_size=32, stride=16, batch_size=2, weight=weight)
        pred_bin = (pred > 0.7).astype(np.uint8)

        # Calculate Metrics
        dice, p, r = compute_metrics(pred_bin, gt_bin)
        
        dice_scores.append(dice)
        precision_scores.append(p)
        recall_scores.append(r)
        
        
        # --- PRINT ALL METRICS HERE ---
        print(f"\nResults for {file_name}:")
        print(f"  > Dice Score:      {dice:.4f}")
        print(f"  > Precision:       {p:.4f}")
        print(f"  > Recall (Sens.):  {r:.4f}")

        # if(dice < 0.5):
        #     return

        if save_vis:
            res_path = os.path.join(RESULTS_DIR, f"sample_{idx}.png")
            visualize_slice(volume, pred_bin, gt_bin, res_path)
            print(f"Visualization saved to: {res_path}")

    # --- FINAL SUMMARY ---
    print("\n" + "="*30)
    print("FINAL GLOBAL AVERAGE RESULTS")
    print("="*30)
    print(f"Mean Dice:      {np.mean(dice_scores):.4f}")
    print(f"Mean Precision: {np.mean(precision_scores):.4f}")
    print(f"Mean Recall:    {np.mean(recall_scores):.4f}")
    print("="*30)

if __name__ == "__main__":
    MODEL_PATH = "/home/vatsal/projects/3D-unet-costa/models/unet3d_20260326_1539/best_model.keras"
    TEST_IMG_DIR = "/home/vatsal/projects/3D-unet-costa/vessel12_split/test/images"
    TEST_MASK_DIR = "/home/vatsal/projects/3D-unet-costa/vessel12_split/test/masks"

    image_files = [f for f in sorted(os.listdir(TEST_IMG_DIR)) if f.endswith(".mhd")]
    test_img = [os.path.join(TEST_IMG_DIR, f) for f in image_files]
    test_mask = [os.path.join(TEST_MASK_DIR, f) for f in image_files]

    model = load_trained_model(MODEL_PATH)
    RESULTS_DIR = "/home/vatsal/projects/3D-unet-costa/results"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Create weight ONCE for the whole evaluation loop
    patch_size = 32
    eval_weight = gaussian_kernel(patch_size, sigma=patch_size//2)

    evaluate_model(model, test_img, test_mask, weight=eval_weight, save_vis=True)