import tensorflow as tf
from tensorflow.keras import layers, models

def unet_3d(input_shape=(128,128,64,1)):
    inputs = layers.Input(input_shape)
    c1 = layers.Conv3D(32, 3, activation='relu', padding='same')(inputs)
    c1 = layers.Conv3D(32, 3, activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling3D(2)(c1)
    c2 = layers.Conv3D(64, 3, activation='relu', padding='same')(p1)
    c2 = layers.Conv3D(64, 3, activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling3D(2)(c2)
    b = layers.Conv3D(128, 3, activation='relu', padding='same')(p2)
    b = layers.Conv3D(128, 3, activation='relu', padding='same')(b)
    u1 = layers.UpSampling3D(2)(b)
    u1 = layers.concatenate([u1, c2])
    c3 = layers.Conv3D(64, 3, activation='relu', padding='same')(u1)
    c3 = layers.Conv3D(64, 3, activation='relu', padding='same')(c3)
    u2 = layers.UpSampling3D(2)(c3)
    u2 = layers.concatenate([u2, c1])
    c4 = layers.Conv3D(32, 3, activation='relu', padding='same')(u2)
    c4 = layers.Conv3D(32, 3, activation='relu', padding='same')(c4)
    outputs = layers.Conv3D(1, 1, activation='sigmoid')(c4)
    return models.Model(inputs, outputs)
