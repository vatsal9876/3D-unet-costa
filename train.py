import os
import numpy as np
from sklearn.model_selection import train_test_split
from model import unet_3d
from dataset import load_dataset
from utils import dice_coef

import tensorflow as tf

# Paths
images_dir = '../data/imagesTr'
labels_dir = '../data/labelsTr'

# Load data (adjust max_samples for full training)
images, masks = load_dataset(images_dir, labels_dir, max_samples=10, shape=(128,128,64))

X_train, X_val, y_train, y_val = train_test_split(images, masks, test_size=0.2, random_state=42)

model = unet_3d(input_shape=(128,128,64,1))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=30, batch_size=1, validation_data=(X_val, y_val))

# Save model
model.save('../model_3d_unet.h5')

# Evaluate Dice on validation set
preds = model.predict(X_val)
dice_scores = [dice_coef(y_true, y_pred > 0.5) for y_true, y_pred in zip(y_val, preds)]
print("Mean Dice coefficient:", np.mean(dice_scores))
