# -------------------------------------------------
# GPU CONFIGURATION (MUST COME BEFORE TENSORFLOW)
# -------------------------------------------------
import os

# Select the NVIDIA GPU (change index if needed)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
import sys
import numpy as np
import datetime

# -------------------------------------------------
# GPU Memory Growth
# -------------------------------------------------
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        print("GPUs detected:", gpus)

    except RuntimeError as e:
        print(e)

else:
    print("⚠️ No GPU detected. Running on CPU.")
    sys.exit()


# -------------------------------------------------
# Allow imports from project root
# -------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from model import unet_3d
from dataset import load_dataset
from utils_old import combined_loss


# -------------------------------------------------
# Dataset Paths
# -------------------------------------------------
images_dir = "/home/vatsal/projects/3D-unet-costa/vessel12_dataset/vessel12_images"
labels_dir = "/home/vatsal/projects/3D-unet-costa/vessel12_dataset/VESSEL12_01-20_Lungmasks"


print("Loading dataset...")
X_train, Y_train, X_test, Y_test = load_dataset(images_dir, labels_dir)


# -------------------------------------------------
# Shuffle Training Data
# -------------------------------------------------
idx = np.random.permutation(len(X_train))

X_train = X_train[idx]
Y_train = Y_train[idx]


print("Train patches:", X_train.shape, Y_train.shape)
print("Validation patches:", X_test.shape, Y_test.shape)


# -------------------------------------------------
# Create Model Directory
# -------------------------------------------------
def create_model_dir(base_path="/home/vatsal/projects/3D-unet-costa/models"):

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")

    model_dir = os.path.join(base_path, f"unet3d_{timestamp}")

    os.makedirs(model_dir, exist_ok=True)

    return model_dir


model_dir = create_model_dir()


# -------------------------------------------------
# Callbacks
# -------------------------------------------------
callbacks = [

    tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(model_dir, "best_model.keras"),
        monitor="val_loss",
        save_best_only=True,
        mode="min",
        verbose=1
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=12,
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
        log_dir=os.path.join(model_dir, "logs"),
        histogram_freq=1
    )
]


# -------------------------------------------------
# Build Model
# -------------------------------------------------
model = unet_3d(input_shape=(64, 64, 64, 1))


model.compile(

    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),

    loss=combined_loss,

    metrics=[
        tf.keras.metrics.Precision(),
        tf.keras.metrics.Recall()
    ]
)

model.summary()


# -------------------------------------------------
# Train Model
# -------------------------------------------------
print("Starting training...")

history = model.fit(

    X_train,
    Y_train,

    validation_data=(X_test, Y_test),

    epochs=100,

    batch_size=1,

    callbacks=callbacks
)


# -------------------------------------------------
# Save Final Model
# -------------------------------------------------
model.save(os.path.join(model_dir, "final_model.keras"))

print("Training complete. Model saved.")