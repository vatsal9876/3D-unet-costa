import tensorflow as tf
import numpy as np
from scipy.ndimage import distance_transform_edt as edt
from scipy.ndimage import binary_erosion

def soft_erode(img):
    """
    3D Morphological Erosion using Min-Pooling.
    Logic: -max_pool3d(-img) is equivalent to min_pooling.
    """
    # Uses a 3x3x3 kernel to find the local minimum
    return -tf.nn.max_pool3d(-img, ksize=[1, 3, 3, 3, 1], strides=[1, 1, 1, 1, 1], padding='SAME')

def soft_skeletonize(x, iterations=3):
    """
    Extracts the 3D centerline/skeleton of the vascular structure.
    """
    for _ in range(iterations):
        eroded = soft_erode(x)
        # Iteratively union the eroded layers to find the 'core' skeleton
        x = tf.keras.layers.Maximum()([x, eroded])
    return x

@tf.function
def evaluate_cldice_score(y_true, y_pred, iterations=3, smooth=1e-6):
    """
    Calculates the clDice score for 3D volumes.
    Inputs:
        y_true, y_pred: 5D Tensors [Batch, Depth, Height, Width, Channels]
    Returns:
        Scalar score between 0.0 and 1.0
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # 1. Extract 3D Skeletons
    skel_true = soft_skeletonize(y_true, iterations)
    skel_pred = soft_skeletonize(y_pred, iterations)

    # 2. Topology Precision: How much of the predicted skeleton is within the true vessel
    tcl_p = (tf.reduce_sum(skel_pred * y_true) + smooth) / (tf.reduce_sum(skel_pred) + smooth)

    # 3. Topology Recall: How much of the true skeleton is within the predicted vessel
    tcl_r = (tf.reduce_sum(skel_true * y_pred) + smooth) / (tf.reduce_sum(skel_true) + smooth)

    # 4. Harmonic Mean (Dice) of Topology Precision and Recall
    return 2.0 * (tcl_p * tcl_r) / (tcl_p + tcl_r + smooth)

def get_3d_cldice(gt, pred):
    """
    Helper function to handle raw 3D inputs (Depth, Height, Width).
    """
    # Ensure inputs are 5D: (1, D, H, W, 1)
    if len(gt.shape) == 3:
        gt = gt[np.newaxis, ..., np.newaxis]
    if len(pred.shape) == 3:
        pred = pred[np.newaxis, ..., np.newaxis]

    # Binarize if necessary (threshold at 0.5)
    # gt = (gt > 0.5).astype(np.float32)
    # pred = (pred > 0.5).astype(np.float32)

    return evaluate_cldice_score(gt, pred).numpy()


def calculate_metrics_safe(y_true, y_pred, spacing=(1.0, 1.0, 1.0)):
    # 1. Force boolean to save space (1 byte per voxel vs 8)
    y_true = np.asanyarray(y_true, dtype=bool)
    y_pred = np.asanyarray(y_pred, dtype=bool)

    # 2. Crop to ROI (The biggest memory saver)
    bbox = get_bbox(y_true, y_pred)
    if bbox is None:
        return {"HD95": np.inf, "ASSD": np.inf}

    y_true_c = y_true[bbox]
    y_pred_c = y_pred[bbox]

    # 3. Get Boundaries
    # Using 'structure' helps scipy optimize the erosion
    gt_b = y_true_c ^ binary_erosion(y_true_c)
    pr_b = y_pred_c ^ binary_erosion(y_pred_c)

    if not np.any(gt_b) or not np.any(pr_b):
        return {"HD95": np.inf, "ASSD": np.inf}

    # 4. Compute first distance map and immediately extract points
    # We cast to float32 to save 50% RAM on the map
    dist_map = edt(~gt_b, sampling=spacing).astype(np.float32)
    d_pred_to_gt = dist_map[pr_b]

    del dist_map # Free memory before next heavy operation

    # 5. Compute second distance map
    dist_map = edt(~pr_b, sampling=spacing).astype(np.float32)
    d_gt_to_pred = dist_map[gt_b]

    del dist_map

    # 6. Final Metric Calculation
    # We don't concatenate large arrays; we use math to find the mean
    hd95 = np.percentile(np.hstack([d_pred_to_gt, d_gt_to_pred]), 95)

    # ASSD is the average of all surface distances
    assd = (np.sum(d_pred_to_gt) + np.sum(d_gt_to_pred)) / (len(d_pred_to_gt) + len(d_gt_to_pred))

    return {"HD95": hd95, "ASSD": assd}

def compute_metrics(pred, gt):
    pred, gt = pred.astype(bool), gt.astype(bool)
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    return dice, precision, recall


def get_bbox(mask_a, mask_b, padding=5):
    """Finds a bounding box surrounding both masks to save memory."""
    combined = mask_a | mask_b
    coords = np.array(np.nonzero(combined))
    if coords.size == 0:
        return None

    mins = np.maximum(coords.min(axis=1) - padding, 0)
    maxs = np.minimum(coords.max(axis=1) + padding, combined.shape)
    return tuple(slice(mins[i], maxs[i]) for i in range(3))
