# -------------------------------------------------
# GPU SETUP
# -------------------------------------------------
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
import numpy as np
import datetime
from sklearn.model_selection import train_test_split


gpus = tf.config.list_physical_devices('GPU')

if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    print("GPU detected:", gpus)
else:
    print("No GPU detected, running on CPU")


# -------------------------------------------------
# PROJECT IMPORTS
# -------------------------------------------------
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from dataset.generator import data_generator
from model import build_unet_3d as unet_3d
from utils.losses import combined_loss, dice_loss   # ✅ FIXED (added dice_loss)


patch_size = 32



# -------------------------------------------------
# DATASET PATHS
# -------------------------------------------------
images_dir = "/home/vatsal/projects/3D-unet-costa/vessel12_split/train_val/images"
labels_dir = "/home/vatsal/projects/3D-unet-costa/vessel12_split/train_val/masks"


# -------------------------------------------------
# GET SCAN PATHS
# -------------------------------------------------
image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(".mhd")])

image_paths = [os.path.join(images_dir, f) for f in image_files]
mask_paths  = [os.path.join(labels_dir, f) for f in image_files]


# -------------------------------------------------
# TRAIN / VALIDATION SPLIT
# -------------------------------------------------
train_img, val_img, train_mask, val_mask = train_test_split(
    image_paths,
    mask_paths,
    test_size=0.2,
    random_state=42
)

print("Train scans:", len(train_img))
print("Val scans:", len(val_img))


# -------------------------------------------------
# GENERATORS
# -------------------------------------------------
train_gen = data_generator(
    train_img,
    train_mask,
    patch_size=patch_size,
    min_vessel_voxels=50
)

val_gen = data_generator(
    val_img,
    val_mask,
    patch_size=patch_size,
    min_vessel_voxels=50
)


# -------------------------------------------------
# TF DATASETS
# -------------------------------------------------
train_dataset = tf.data.Dataset.from_generator(
    lambda: train_gen,
    output_signature=(
        tf.TensorSpec(shape=(None,32,32,32,1), dtype=tf.float32),
        tf.TensorSpec(shape=(None,32,32,32,1), dtype=tf.float32)
    )
)

val_dataset = tf.data.Dataset.from_generator(
    lambda: val_gen,
    output_signature=(
        tf.TensorSpec(shape=(None,32,32,32,1), dtype=tf.float32),
        tf.TensorSpec(shape=(None,32,32,32,1), dtype=tf.float32)
    )
)


# ❌ REMOVED .batch(1) → generator already returns batch
# train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
# val_dataset   = val_dataset.prefetch(tf.data.AUTOTUNE)


# -------------------------------------------------
# CREATE MODEL DIRECTORY
# -------------------------------------------------
def create_model_dir(base_path="/home/vatsal/projects/3D-unet-costa/models"):

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")

    model_dir = os.path.join(base_path, f"unet3d_{timestamp}")

    os.makedirs(model_dir, exist_ok=True)

    return model_dir


model_dir = create_model_dir()


# -------------------------------------------------
# CALLBACKS
# -------------------------------------------------
callbacks = [

    tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(model_dir,"best_model.keras"),
        monitor="val_loss",
        save_best_only=True,
        mode="min",
        verbose=1
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    ),

    tf.keras.callbacks.TensorBoard(
        log_dir=os.path.join(model_dir,"logs"),
        histogram_freq=1
    )
]


# -------------------------------------------------
# BUILD MODEL
# -------------------------------------------------
model = unet_3d(
    input_shape=(32,32,32,1),
    base_filters=16
)


model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss=combined_loss,
    metrics=[
        dice_loss,
        tf.keras.metrics.Precision(thresholds=0.5),
        tf.keras.metrics.Recall(thresholds=0.5)
    ]
)


model.summary()


# -------------------------------------------------
# TRAIN MODEL
# -------------------------------------------------
print("Starting training...")


history = model.fit(

    train_dataset,

    validation_data=val_dataset,

    steps_per_epoch=30,

    validation_steps=20,

    epochs=300,

    callbacks=callbacks
)


# -------------------------------------------------
# SAVE FINAL MODEL
# -------------------------------------------------
model.save(os.path.join(model_dir,"final_model.keras"))

print("Training complete")