import tensorflow as tf
from tensorflow.keras import layers


def conv_block(x, f):
    shortcut = x

    x = layers.Conv3D(f, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv3D(f, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)

    if shortcut.shape[-1] != f:
        shortcut = layers.Conv3D(f, 1, padding='same')(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    return x


def build_unet_3d(depth=64, height=96, width=96,
                  base_filters=32):

    input_shape = (depth, height, width, 1)
    inputs = layers.Input(shape=input_shape)

    f = base_filters

    # -------- Encoder --------
    c1 = conv_block(inputs, f)
    p1 = layers.MaxPooling3D(pool_size=(2,2,2))(c1)

    c2 = conv_block(p1, f*2)
    p2 = layers.MaxPooling3D(pool_size=(2,2,2))(c2)

    c3 = conv_block(p2, f*4)
    p3 = layers.MaxPooling3D(pool_size=(2,2,2))(c3)

    # -------- Bottleneck --------
    bn = conv_block(p3, f*8)

    # -------- Decoder --------
    u3 = layers.UpSampling3D(size=(2,2,2))(bn)
    u3 = layers.Concatenate()([u3, c3])
    c4 = conv_block(u3, f*4)

    u2 = layers.UpSampling3D(size=(2,2,2))(c4)
    u2 = layers.Concatenate()([u2, c2])
    c5 = conv_block(u2, f*2)

    u1 = layers.UpSampling3D(size=(2,2,2))(c5)
    u1 = layers.Concatenate()([u1, c1])
    c6 = conv_block(u1, f)

    outputs = layers.Conv3D(1, 1, activation='sigmoid')(c6)

    return tf.keras.Model(inputs, outputs)