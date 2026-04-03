import tensorflow as tf

def soft_erode(img):
    """
    3D Morphological erosion using min-pooling.
    Handles Rank 5 tensors: [Batch, Depth, Height, Width, Channels]
    """
    # We negate to use max_pool as a min_pool
    # ksize and strides for 3D: [batch, depth, rows, cols, channels]
    p1 = -tf.nn.max_pool3d(-img, ksize=[1, 3, 1, 1, 1], strides=[1, 1, 1, 1, 1], padding='SAME')
    p2 = -tf.nn.max_pool3d(-img, ksize=[1, 1, 3, 1, 1], strides=[1, 1, 1, 1, 1], padding='SAME')
    p3 = -tf.nn.max_pool3d(-img, ksize=[1, 1, 1, 3, 1], strides=[1, 1, 1, 1, 1], padding='SAME')
    return tf.math.minimum(tf.math.minimum(p1, p2), p3)

def soft_skeletonize(x, iterations=3):
    """3D Iterative thinning to find vessel centerlines."""
    for _ in range(iterations):
        eroded = soft_erode(x)
        # Dilate using 3D max pool
        dilation = tf.nn.max_pool3d(eroded, ksize=[1, 3, 3, 3, 1], strides=[1, 1, 1, 1, 1], padding='SAME')
        x = tf.nn.relu(x - dilation) + eroded
    return x

def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_pred = tf.reshape(tf.cast(y_pred, tf.float32), [-1])
    intersection = tf.reduce_sum(y_true * y_pred)
    return 1 - (2. * intersection + smooth) / (
        tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth
    )

def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # Keras binary_crossentropy works on the last dimension by default
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)

    # Ensure BCE has the same rank as y_true for multiplication
    if len(bce.shape) < len(y_true.shape):
        bce = tf.expand_dims(bce, axis=-1)

    p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
    return alpha * tf.pow(1 - p_t, gamma) * bce

def cldice_loss(y_true, y_pred, iterations=3, smooth=1e-6):
    """Calculates the 3D centerline dice for connectivity."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    skel_true = soft_skeletonize(y_true, iterations)
    skel_pred = soft_skeletonize(y_pred, iterations)

    t_prec = (tf.reduce_sum(skel_pred * y_true) + smooth) / (tf.reduce_sum(skel_pred) + smooth)
    t_sens = (tf.reduce_sum(skel_true * y_pred) + smooth) / (tf.reduce_sum(skel_true) + smooth)

    return 1.0 - (2.0 * t_prec * t_sens / (t_prec + t_sens + smooth))

def combined_loss(y_true, y_pred):
    """
    Triple Loss for OpsTwin:
    Volume (Dice) + Imbalance (Focal) + Connectivity (clDice)
    """
    d_loss = dice_loss(y_true, y_pred)
    f_loss = tf.reduce_mean(focal_loss(y_true, y_pred))
    c_loss = cldice_loss(y_true, y_pred)

    return d_loss + 3.3*f_loss + c_loss