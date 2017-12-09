import numpy as np
import tensorflow as tf


def _tfshift(matrix, n, axis=1, invert=False):
    """Handler for shifting one axis at a time.
    Helpful for fftshift if invert is False and ifftshift otherwise
    """
    if invert:
        mid = n - (n + 1) // 2
    else:
        mid = (n + 1) // 2
    if axis == 1:
        start = [0, 0, 0, mid]
        end = [-1, -1, -1, mid]
    else:
        start = [0, 0, mid, 0]
        end = [-1, -1, mid, -1]
    out = tf.concat([tf.slice(matrix, start, [-1, -1, -1, -1]),
                     tf.slice(matrix, [0, 0, 0, 0], end)], axis + 2)
    return out


def tf_fftshift(matrix, n):
    """Performs similar function to numpy's fftshift
    Note: Takes image as a channel first numpy array of shape:
        (batch_size, channels, height, width)
    """
    mat = _tfshift(matrix, n, 1)
    mat2 = _tfshift(mat, n, 0)
    return mat2


def tf_ifftshift(matrix, n):
    """Performs similar function to numpy's ifftshift
    Note: Takes image as a channel first numpy array of shape:
        (batch_size, channels, height, width)
    """
    mat = _tfshift(matrix, n, 1, invert=True)
    mat2 = _tfshift(mat, n, 0, invert=True)
    return mat2


def spectral_pool(image, filter_size=3,
                  return_fft=False):
    """ Perform a single spectral pool operation.
    Args:
        image: numpy array representing an image, channels last
            shape: (batch_size, height, width, channel)
        filter_size: the final dimension of the filter required
        return_fft: bool, if True function also returns the raw
                          fourier transform
    Returns:
        An image of same shape as input
    NOTE: Filter size is enforced to be odd here. It is required to
    prevent the need for treating edge cases
    """
    # filter size should always be odd:
    assert filter_size % 2

    tf.reset_default_graph()
    im = tf.placeholder(shape=image.shape, dtype=tf.float32)
    dim = im.get_shape().as_list()[1]

    # make channels first & get fft
    im_channel_first = tf.transpose(im, perm=[0, 3, 1, 2])
    im_fft = tf.fft2d(tf.cast(im_channel_first, tf.complex64))

    # shift the image and crop based on the bounding box:
    im_fshift = tf_fftshift(im_fft, dim)

    # make channels last as required by crop function
    im_channel_last = tf.transpose(im_fshift, perm=[0, 2, 3, 1])

    offset = int(dim / 2) - int(filter_size / 2)
    im_cropped = tf.image.crop_to_bounding_box(im_channel_last, offset, offset,
                                               filter_size, filter_size)

    # pad zeros to the image
    # this is required only when we're visualizing the image and not in
    # the final spectral layer
    # required to handle odd and even image size
    offset = int((dim + 1 - filter_size) / 2)
    im_pad = tf.image.pad_to_bounding_box(im_cropped, offset, offset, dim, dim)
    # im_pad = im_cropped

    # perform ishift and take the inverse fft and throw img part
    # make channels first for ishift and ifft2d:
    im_channel_first2 = tf.transpose(im_pad, perm=[0, 3, 1, 2])
    im_ishift = tf_ifftshift(im_channel_first2, dim)
    im_channel_last = tf.transpose(tf.real(tf.ifft2d(im_ishift)),
                                   perm=[0, 2, 3, 1])

    # normalize image:
    channel_max = tf.reduce_max(im_channel_last, axis=(0, 1, 2))
    channel_min = tf.reduce_min(im_channel_last, axis=(0, 1, 2))
    im_out = tf.divide(im_channel_last - channel_min,
                       channel_max - channel_min)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        if return_fft:
            im_fftout, im_new = sess.run([im_fft, im_out],
                                         feed_dict={im: image})
            return im_fftout, im_new
        else:
            im_new = sess.run([im_out],
                               feed_dict={im: image})
            return im_new


def max_pool(image, pool_size=2):
    """ Perform a single max pool operation.
    Args:
        image: numpy array representing an image
            shape: (num_images, height, width, channel)
        pool_size: number of dimensions to throw away in each dimension,
                   same as the filter size of max_pool
    Returns:
        An image of shape (n, n, 1) if grayscale is True or same as input
    """
    imsize = image.shape[1]

    im_channel_last = np.moveaxis(image, 0, 2)
    im_new = im_channel_last.copy()
    for i in range(0, imsize, pool_size):
        for j in range(0, imsize, pool_size):
            max_val = np.max(im_channel_last[i: i + pool_size,
                                             j: j + pool_size],
                                             axis=(0, 1))
            im_new[i: i + pool_size, j: j + pool_size] = max_val
    im_new = np.moveaxis(im_new, 2, 0)
    return im_new


def l2_loss_images(orig_images, mod_images):
    """Calculates the loss for a set of modified images vs original
    formular: l2(orig-mod)/l2(orig)
    Args:
        orig_images: numpy array size (batch, dims..)
        mod_images: numpy array of same dim as orig_images
    Returns:
        single value, i.e. loss
    """
    n = orig_images.shape[0]
    # convert to 2d:
    oimg = orig_images.reshape(n, -1)
    mimg = mod_images.reshape(n, -1)

    # bring to same scale if not scales already
    if oimg.max() > 2:
        oimg = oimg / 255.
    if mimg.max() > 2:
        mimg = mimg / 255.

    error_norm = np.linalg.norm(oimg - mimg, axis=0)
    base_norm = np.linalg.norm(oimg, axis=0)
    return np.mean(error_norm / base_norm)
