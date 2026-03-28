import tensorflow as tf

def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])

    intersection = tf.reduce_sum(y_true * y_pred)
    return 1 - (2. * intersection + smooth) / (
        tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth
    )

def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # this already keeps channel dimension correctly
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)

    # ensure SAME rank as y_true (not blindly expanding)
    if len(bce.shape) < len(y_true.shape):
        bce = tf.expand_dims(bce, axis=-1)

    p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)

    return alpha * tf.pow(1 - p_t, gamma) * bce


def combined_loss(y_true, y_pred):
    return dice_loss(y_true, y_pred) + tf.reduce_mean(focal_loss(y_true, y_pred))