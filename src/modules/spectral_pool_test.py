"""NOTE: THIS CODE ONLY WORKS FOR A SINGLE IMAGE.
TO POOL MULTIPLE IMAGES TOGETHER, CHECKOUT spectral_pool.py
"""
import numpy as np
import tensorflow as tf
from PIL import Image


def tfshift(matrix, n, axis=1):
    mid = (n + 1) // 2
    if axis == 1:
        start = [0, 0, mid]
        end = [-1, -1, mid]
    else:
        start = [0, mid, 0]
        end = [-1, mid, -1]
    out = tf.concat([tf.slice(matrix, start, [-1, -1, -1]),
                     tf.slice(matrix, [0, 0, 0], end)], axis + 1)
    return out


def tfroll(matrix, n):
    mat = tfshift(matrix, n, 1)
    print(mat.get_shape())
    mat2 = tfshift(mat, n, 0)
    return mat2


def get_low_pass_filter(shape, pool_size):
    lowpass = np.zeros(shape=shape, dtype=np.float32)
    cutoff_freq = int((shape[1] / (pool_size * 2)))
    lowpass[:, :cutoff_freq, :cutoff_freq] = 1
    lowpass[:, :cutoff_freq, -cutoff_freq:] = 1
    lowpass[:, -cutoff_freq:, :cutoff_freq] = 1
    lowpass[:, -cutoff_freq:, -cutoff_freq:] = 1
    return lowpass


def spectral_pool(image, pool_size=4,
                  convert_grayscale=True):
    """ Perform a single spectral pool operation.
    Args:
        image: numpy array representing an image
        pool_size: number of dimensions to throw away in each dimension,
                   same as the filter size of max_pool
        convert_grayscale: bool, if True, the image will be converted to
                           grayscale
    Returns:
        An image of shape (n, n, 1) if grayscale is True or same as input
    """
    tf.reset_default_graph()
    im = tf.placeholder(shape=image.shape, dtype=tf.float32)
    if convert_grayscale:
        im_conv = tf.image.rgb_to_grayscale(im)
    else:
        im_conv = im
    # make channels first
    im_channel_first = tf.transpose(im_conv, perm=[2, 0, 1])
    im_fft = tf.fft2d(tf.cast(im_channel_first, tf.complex64))
    lowpass = tf.get_variable(name='lowpass',
                              initializer=get_low_pass_filter(
                                    im_channel_first.get_shape().as_list(),
                                    pool_size))
    im_magnitude = tf.multiply(tf.abs(im_fft), lowpass)
    im_angles = tf.angle(im_fft)
    part1 = tf.complex(real=im_magnitude,
                       imag=tf.zeros_like(im_angles))
    part2 = tf.exp(tf.complex(real=tf.zeros_like(im_magnitude),
                              imag=im_angles))
    im_fft_lowpass = tf.multiply(part1, part2)
    im_transformed = tf.ifft2d(im_fft_lowpass)
    # make channels last and real values:
    im_channel_last = tf.real(tf.transpose(im_transformed, perm=[1, 2, 0]))

    # normalize image:
    channel_max = tf.reduce_max(im_channel_last, axis=(0, 1))
    channel_min = tf.reduce_min(im_channel_last, axis=(0, 1))
    im_out = tf.divide(im_channel_last - channel_min,
                       channel_max - channel_min)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        im_fftout, im_new = sess.run([im_magnitude, im_out],
                                     feed_dict={im: image})

    return im_fftout, im_new


def max_pool(image, pool_size=4,
             convert_grayscale=True):
    """ Perform a single max pool operation.
    Args:
        image: numpy array representing an image
        pool_size: number of dimensions to throw away in each dimension,
                   same as the filter size of max_pool
        convert_grayscale: bool, if True, the image will be converted to
                           grayscale
    Returns:
        An image of shape (n, n, 1) if grayscale is True or same as input
    """
    if convert_grayscale:
        im = Image.fromarray(np.uint8(image * 255)).convert('F')
    else:
        im = im = Image.fromarray(np.uint8(image * 255))
    im_np = np.asarray(im)
    im_np = np.atleast_3d(im_np)
    imsize = im_np.shape[0]

    im_new = im_np.copy()
    for i in range(0, imsize, pool_size):
        for j in range(0, imsize, pool_size):
            max_val = np.max(im_new[i: i + pool_size, j: j + pool_size, :],
                             axis=(0, 1))
            im_new[i: i + pool_size, j: j + pool_size, :] = max_val
    return im_new


def get_fft_plot(fft, shift_channel=True, eps=1e-12):
    """ Convert a fourier transform returned from tensorflow in a format
    that can be plotted.
    Args:
        fft: numpy array with image and channels
        shift_channel: if True, the channels are assumed as first dimension and
                       will be moved to the end.
        eps: to be added before taking log
    """
    if shift_channel:
        fft = np.squeeze(np.moveaxis(np.absolute(fft), 0, -1))
    fft = np.log(fft + eps)
    mn = np.min(fft, axis=(0, 1))
    mx = np.max(fft, axis=(0, 1))
    fft = (fft - mn) / (mx - mn)
    return np.fft.fftshift(fft)
