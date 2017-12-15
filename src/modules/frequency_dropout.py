"""Perform frequency dropout."""
import numpy as np
import tensorflow as tf


def _frequency_dropout_mask(height, frequency_to_truncate_above):
    """Create a mask to be used for frequency dropout.

    Args:
        height: int, the height of the image to create a mask for.
            For a 32x32 image, this should be 32.
        frequency_to_truncate_above: Tensor of shape (,) (i.e. scalar). All
            frequencies above this will be set to zero. For an image with
            a height of 32, a number above 16 will have no effect. For an
            image with a height of 31, an input above 15 will have no effect.

    Returns:
        dropout_mask: Tensor of shape (height, height)
            The result can be multiplied by the FFT of an image to create
            a modified FFT where all frequencies above the cutoff have
            been set to zero. Therefore, the value of the mask will be 1
            for the frequencies below the truncation level, and 0 for the
            frequencies above it. In other words, it is really the mask
            of values to retain, not the mask of values to drop.
    """
    cutoff_shape = frequency_to_truncate_above.get_shape().as_list()
    assert len(cutoff_shape) == 0

    # Create an array "indexes" of the form [0 1 2 3 3 2 1], if height = 7,
    # or [0 1 2 3 4 3 2 1], if height = 8
    mid = height // 2
    if height % 2 == 1:
        go_to = mid + 1
    else:
        go_to = mid
    indexes = np.concatenate((
        np.arange(go_to),
        np.arange(mid, 0, -1)
    )).astype(np.float32)

    # Create a matrix "highest_frequency" of the form
    # [[ 0.  1.  2.  3.  3.  2.  1.]
    #  [ 1.  1.  2.  3.  3.  2.  1.]
    #  [ 2.  2.  2.  3.  3.  2.  2.]
    #  [ 3.  3.  3.  3.  3.  3.  3.]
    #  [ 3.  3.  3.  3.  3.  3.  3.]
    #  [ 2.  2.  2.  3.  3.  2.  2.]
    #  [ 1.  1.  2.  3.  3.  2.  1.]]
    # if height = 7, or
    #
    # [[ 0.  1.  2.  3.  4.  3.  2.  1.]
    #  [ 1.  1.  2.  3.  4.  3.  2.  1.]
    #  [ 2.  2.  2.  3.  4.  3.  2.  2.]
    #  [ 3.  3.  3.  3.  4.  3.  3.  3.]
    #  [ 4.  4.  4.  4.  4.  4.  4.  4.]
    #  [ 3.  3.  3.  3.  4.  3.  3.  3.]
    #  [ 2.  2.  2.  3.  4.  3.  2.  2.]
    #  [ 1.  1.  2.  3.  4.  3.  2.  1.]]
    #
    # if height = 8.
    xs = np.broadcast_to(indexes, (height, height))
    ys = np.broadcast_to(np.expand_dims(indexes, -1), (height, height))
    highest_frequency = np.maximum(xs, ys)

    comparison_mask = tf.constant(highest_frequency)
    dropout_mask = tf.cast(tf.less_equal(
        comparison_mask,
        frequency_to_truncate_above
    ), tf.complex64)
    return dropout_mask


def test_frequency_dropout(images, frequency_to_truncate_above):
    """Demonstrate the use of _frequency_dropout_mask.

    Args:
        images: ndarray of shape (num_images, num_channels, height, width)
        frequency_to_truncate_above: Tensor of shape (,) (i.e. scalar). All
            frequencies above this will be set to zero. For an image with
            a height of 32, a number above 16 will have no effect. For an
            image with a height of 31, an input above 15 will have no effect.

    Returns:
        downsampled_images: ndarray of shape (num_images, num_channels,
            height, widtdh).
    """
    assert len(images.shape) == 4
    N, C, H, W = images.shape
    assert H == W
    frq_drop_mask = _frequency_dropout_mask(H, frequency_to_truncate_above)
    tf_images = tf.constant(images, dtype=tf.complex64)
    images_fft = tf.fft2d(tf_images)
    images_trunc = images_fft * frq_drop_mask
    images_back = tf.real(tf.ifft2d(images_trunc))
    with tf.Session() as sess:
        downsampled_images = sess.run(images_back)

    return downsampled_images
