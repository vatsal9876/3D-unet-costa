# 3D U-Net for Cerebrovascular Segmentation (ADAM Subset - COSTA)

## What this project does (non-technical)

This project builds a deep learning system that can automatically detect and segment blood vessels in 3D brain scans. Instead of analyzing single 2D slices, the model processes full volumetric data, allowing it to capture complex vessel structures and continuity.

It is specifically designed to handle thin, branching vessels, which are difficult to detect due to:

* Extreme class imbalance (very few vessel voxels vs background)
* Fragile connectivity (breaks in vessels reduce usefulness)
* High structural complexity

The system is trained and evaluated on a subset of the ADAM dataset (COSTA benchmark), focusing on accurate and structurally consistent vessel extraction.

---
## Usage

### 1. Clone the repository
git clone https://github.com/your-username/repo-name.git
cd repo-name

### 2. Install dependencies
pip install -r requirements.txt

### 3. Dataset setup
This project uses the **COSTA dataset** (not included in the repository).

- Download the dataset from: <link>
- Update the dataset paths inside the code before running (currently hardcoded)

### 4. Training
python train.py

### 5. Inference
python infer.py

---

## Key highlights

* 3D volumetric segmentation (captures full vessel continuity, unlike 2D slice models)
* Vessel-aware patch sampling to overcome extreme class imbalance
* Hybrid loss (Dice + Focal + clDice) targeting accuracy + hard voxels + topology
* Residual 3D U-Net for stable deep training
* Strong geometric evaluation (HD95, ASSD) beyond overlap metrics

---

## Dataset

* Source: ADAM subset (COSTA)
* Format: 3D NIfTI volumes (.nii.gz)
* Input: Brain scans
* Output: Binary vessel masks

### Preprocessing

* Axis alignment: (H, W, D) → (D, H, W)
* Intensity normalization (z-score)
* Mask binarization

---

## Data Sampling Strategy

Naive random sampling fails due to extreme imbalance.

This project uses **biased patch extraction**:

* Patch size: **64 × 96 × 96** (D × H × W)
* 70% probability → vessel-centered patches
* Remaining → background patches
* Constraints:

  * Minimum vessel voxels per patch
  * Background threshold control

Impact:

* Prevents model collapse to predicting all background
* Forces learning of vessel structures

---

## Model Architecture

### Base: 3D U-Net with Residual Blocks

**Input:** (64, 96, 96, 1)

### Encoder

* Conv Block (32 filters)
* Conv Block (64 filters)
* Conv Block (128 filters)
* Downsampling via MaxPool3D

### Bottleneck

* Conv Block (256 filters)

### Decoder

* Upsampling + skip connections
* Conv Blocks mirroring encoder

### Output

* 1×1×1 Conv3D
* Sigmoid activation (voxel-wise probability)

### Residual Conv Block

Each block:

* Conv3D → BN → ReLU
* Conv3D → BN
* Shortcut connection
* Addition + ReLU

Why this matters:

* Stabilizes training
* Helps gradient flow in deep 3D networks

---

## Loss Function (core strength)

### Combined Loss

```
Loss = Dice + 3.3 × Focal + clDice
```

### Components

**1. Dice Loss (Region Accuracy)**

* Measures overlap between prediction and ground truth
* Handles imbalance better than BCE

**2. Focal Loss (Hard Example Mining)**

* Focuses on difficult voxels
* Reduces dominance of easy background

**3. clDice (Connectivity-aware)**

* Uses soft skeletonization
* Preserves vessel topology
* Penalizes broken vessels

Why this combination works:

* Dice → global overlap
* Focal → pixel-level difficulty
* clDice → structural correctness

---

## Training Setup

* Optimizer: Adam
* Patch-based generator (on-the-fly)
* Batch size: 1 (due to 3D memory constraints)
* Train/Validation split: 80/20

---

## Evaluation Metrics (Test Results)

### Final Global Results

* **Dice:** 0.8700 → strong volumetric overlap
* **Precision:** 0.8368 → moderate false positives
* **Recall:** 0.9097 → most vessels are detected
* **HD95:** 3.6719 → low worst-case boundary error
* **ASSD:** 0.7675 → consistent surface accuracy
* **clDice:** 0.8700 → vessel connectivity preserved
---

## Qualitative Results (what it actually looks like)

> Add images in `/results/` and reference them here.

### Slice-wise comparison

* Input slice
* Ground truth mask
* Predicted mask

What to look for:

* Thin vessels retained (no breaks)
* Minimal background noise
* Smooth boundaries (no jagged artifacts)

### 3D consistency (important)

* Vessels remain continuous across adjacent slices
* No sudden disappear/reappear artifacts

---

## Failure Cases (real weaknesses)

### Case 1: False positives in high-intensity regions

* Cause: intensity similarity with vessels
* Symptom: small noisy blobs
* Fix: connected component filtering / attention

### Case 2: Missed ultra-thin vessels

* Cause: resolution + patch sampling limits
* Symptom: broken micro-branches
* Fix: higher resolution patches / multi-scale training

### Case 3: Boundary over-segmentation

* Cause: high recall bias (Focal + Dice)
* Symptom: slightly thicker vessels than GT
* Fix: add boundary-aware loss or calibration

---

## Core strengths

* Robust handling of extreme class imbalance via vessel-focused sampling
* Explicit preservation of vessel connectivity using clDice
* Full 3D spatial modeling for continuous structure learning
* Evaluation includes both overlap and geometric surface metrics

Result: produces structurally consistent and usable vessel segmentations.

---

## Limitations (real ones, not fluff)

* Precision < Recall → false positives still present
* Patch-based training limits global context awareness
* Batch size = 1 → noisy gradients, slower convergence

---

## High-impact improvements (if you extend this)

* Attention U-Net / spatial attention → reduce false positives
* Multi-scale training (coarse + fine patches)
* Post-processing: connected component filtering to remove noise
* Test-time augmentation for stability boost

---

## Tech Stack

* TensorFlow / Keras
* NumPy, Nibabel
* Scikit-learn
* Google Colab (training)

---

## Bottom line

This is a **topology-aware 3D segmentation pipeline**, not a basic U-Net implementation. It explicitly addresses the three hard parts of vessel segmentation: imbalance, thin structures, and connectivity — and backs it with geometric evaluation, not just overlap scores.
