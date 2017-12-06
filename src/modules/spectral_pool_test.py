import numpy as np
import tensorflow as tf


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
    cutoff_freq = int((shape[1] - pool_size) / 4)
    lowpass[:, :cutoff_freq, :cutoff_freq] = 1
    lowpass[:, :cutoff_freq, -cutoff_freq:] = 1
    lowpass[:, -cutoff_freq:, :cutoff_freq] = 1
    lowpass[:, -cutoff_freq:, -cutoff_freq:] = 1
    print(cutoff_freq, np.sum(lowpass)/256/256)
    return lowpass


def spectral_pool(image, pool_size=4,
                  convert_grayscale=True):
    """ Perform a single spectral pool operation.
    Args:
        image: 2D image, same height and width
        pool_size: number of dimensions to throw away in each dimension
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
    im_magnitude = tf.abs(im_fft)
    im_angles = tf.angle(im_fft)
    lowpass = tf.get_variable(name='lowpass',
                              initializer=get_low_pass_filter(
                                    im_channel_first.get_shape().as_list(),
                                    pool_size))
    # im_fft_roll = tfroll(im_fft, n)
    # print(im_fft_roll.get_shape())
    # target_size = n - pool_size
    # # crop the extra dimensions centrally
    # im_crop = tf.image.resize_image_with_crop_or_pad(image=im_fft_roll,
    #                                                  target_height=target_size,
    #                                                  target_width=target_size)
    # # pad with 0s to get the original size
    # im_pad = tf.image.resize_image_with_crop_or_pad(image=im_crop,
    #                                                  target_height=n,
    #                                                  target_width=n)
    # n = im_pad.get_shape().as_list()[0]
    # im_unroll = tfroll(im_pad, n)
    # print(im_unroll.get_shape())
    part1 = tf.complex(real=tf.multiply(im_magnitude, lowpass),
                       imag=tf.zeros_like(im_angles))
    part2 = tf.exp(tf.complex(real=tf.zeros_like(im_magnitude),
                              imag=im_angles))
    im_fft_lowpass = tf.multiply(part1, part2)
    im_transformed = tf.ifft2d(im_fft_lowpass)
    # make channels last:
    im_out = tf.transpose(im_transformed, perm=[1, 2, 0])
    # im_transformed = tf.ifft(im_fft)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        lp, im_fftout, im_new = sess.run([im_fft_lowpass,im_fft, im_out],
                                     feed_dict={im: image})

    return lp, im_fftout, im_new


def max_pool(image, pool_size=4,
             convert_grayscale=True):
    im = image.convert('F')
    im_np = np.asarray(im)
    imsize = im_np.shape[0]

    im_new = im_np.copy()
    for i in range(0, imsize, pool_size):
        for j in range(0, imsize, pool_size):
            max_val = np.max(im_new[i: i + pool_size, j: j + pool_size])
            im_new[i: i + pool_size, j: j + pool_size] = max_val
    return im_new
