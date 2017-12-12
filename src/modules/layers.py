from .spectral_pool import _common_spectral_pool
import numpy as np
import tensorflow as tf


class default_conv_layer(object):
    def __init__(self, input_x, in_channel, out_channel,
                 kernel_shape, rand_seed, m=0):
        """
        NOTE: Image should be CHANNEL FIRST
        :param input_x: Should be a 4D array like:
                            (batch_num, channel_num, img_len, img_len)
        :param in_channel: The number of channels
        :param out_channel: number of filters required
        :param kernel_shape: kernel size
        :param rand_seed: random seed
        :param index: The layer index used for naming
        """
        assert len(input_x.shape) == 4
        assert input_x.shape[2] == input_x.shape[3]
        assert input_x.shape[1] == in_channel

        # Alternative using layers but not using it
        # with tf.variable_scope('conv_layer_{0}'.format(m)):
        # conv_out = tf.layers.conv2d(
        #                 inputs=input_x,
        #                 filters=out_channel,
        #                 kernel_size=kernel_shape,
        #                 strides=(1, 1),
        #                 padding='valid',
        #                 data_format='channels_first',
        #                 activation=tf.nn.relu,
        #                 use_bias=True,
        #                 kernel_initializer=tf.glorot_uniform_initializer(
        #                                                 seed=rand_seed),
        #                 bias_initializer=tf.glorot_uniform_initializer(
        #                                                 seed=rand_seed),
        #                 name='conv_layer_{0}'.format(m),
        #                 trainable=True
        #                             )
        with tf.variable_scope('conv_layer_{0}'.format(m)):
            with tf.name_scope('conv_kernel'):
                w_shape = [kernel_shape, kernel_shape, in_channel, out_channel]
                weight = tf.get_variable(
                     name='conv_kernel_{0}'.format(m),
                     shape=w_shape,
                     initializer=tf.glorot_uniform_initializer(seed=rand_seed))
                self.weight = weight

            with tf.variable_scope('conv_bias'):
                b_shape = [out_channel]
                bias = tf.get_variable(
                   name='conv_bias_{0}'.format(m), shape=b_shape,
                   initializer=tf.glorot_uniform_initializer(seed=rand_seed))
                self.bias = bias

            # strides [1, x_movement, y_movement, 1]
            conv_out = tf.nn.conv2d(input_x, weight,
                                    strides=[1, 1, 1, 1],
                                    padding="SAME",
                                    data_format="NCHW")
            cell_biased = tf.nn.bias_add(
                conv_out,
                self.bias, data_format='NCHW'
            )
            cell_out = tf.nn.relu(cell_biased)

            self.cell_out = cell_out

            tf.summary.histogram('conv_layer/{}/kernel'.format(m), weight)
            tf.summary.histogram('conv_layer/{}/bias'.format(m), bias)
            self.cell_out = conv_out

    def output(self):
        return self.cell_out


class fc_layer(object):
    def __init__(self, input_x, in_size, out_size, rand_seed,
                 activation_function=None, m=0):
        """
        :param input_x: The input of the FC layer. It should be a flatten vector.
        :param in_size: The length of input vector.
        :param out_size: The length of output vector.
        :param rand_seed: An integer that presents the random seed used to generate the initial parameter value.
        :param keep_prob: The probability of dropout. Default set by 1.0 (no drop-out applied)
        :param activation_function: The activation function for the output. Default set to None.
        :param index: The index of the layer. It is used for naming only.

        """
        with tf.variable_scope('fc_layer_{0}'.format(m)):
            with tf.name_scope('fc_kernel'):
                w_shape = [in_size, out_size]
                weight = tf.get_variable(
                     name='fc_kernel_{0}'.format(m),
                     shape=w_shape,
                     initializer=tf.glorot_uniform_initializer(seed=rand_seed))
                self.weight = weight

            with tf.variable_scope('fc_kernel'):
                b_shape = [out_size]
                bias = tf.get_variable(
                       name='fc_bias_{0}'.format(m),
                       shape=b_shape,
                       initializer=tf.glorot_uniform_initializer(
                                                            seed=rand_seed))
                self.bias = bias

            cell_out = tf.add(tf.matmul(input_x, weight), bias)
            if activation_function is not None:
                cell_out = activation_function(cell_out)

            self.cell_out = cell_out

    def output(self):
        return self.cell_out


class spectral_pool_layer(object):
    def __init__(self, input_x, filter_size=3, freq_dropout=0,
                 m=0, train_phase=True):
        """ Perform a single spectral pool operation.
        Args:
            input_x: numpy array representing an image, channels last
                shape: (batch_size, height, width, channel)
            filter_size: int, the final dimension of the filter required
            freq_dropout: int, clip freq number
        Returns:
            An image of similar shape as input after reduction
        NOTE: Filter size is enforced to be odd here. It is required to
        prevent the need for treating edge cases
        """
        # filter size should always be odd:
        assert filter_size % 2
        # assert only 1 dimension passed for filter size
        assert isinstance(filter_size, int)
        # assign var:
        self.freq_dropout = freq_dropout

        with tf.variable_scope('spectral_pool_layer_{0}'.format(m)):
            im_fft = tf.fft2d(tf.cast(input_x, tf.complex64))
            im_transformed = _common_spectral_pool(im_fft, filter_size)
            im_out = tf.real(tf.ifft2d(im_transformed))

        # THERE COULD BE A NORMALISING STEP HERE SIMILAR TO BATCH NORM BUT
        # I'M SKIPPING IT HERE
        # im_channel_last = tf.transpose(tf.real(tf.ifft2d(im_ishift)),
        #                                perm=[0, 2, 3, 1])

        # # normalize image:
        # channel_max = tf.reduce_max(im_channel_last, axis=(0, 1, 2))
        # channel_min = tf.reduce_min(im_channel_last, axis=(0, 1, 2))
        # im_out = tf.divide(im_channel_last - channel_min,
        #                    channel_max - channel_min)
        self.cell_out = im_out

    def output(self):
        return self.cell_out

    def _freq_dropout_matrix(self):
        """Create a matrix to be multiplied to implement freq dropout.
        Its a 1s matrix with values after freq_dropout made 0"""
        fft_shape = self.fft_shape
        freq_dropout = self.freq_dropout
        out = np.zeros(shape=fft_shape[1:],
                       dtype=np.complex64)
        start = int((fft_shape[-1] - freq_dropout)/2)
        end = freq_dropout + start
        out[:, start:end, start:end] = 1
        return out

    def _no_dropout_matrix(self):
        return np.ones(shape=self.fft_shape[1:],
                       dtype=np.complex64)

    def _tfshift(self, matrix, n, axis=1, invert=False):
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

    def _tf_fftshift(self, matrix, n):
        """Performs similar function to numpy's fftshift
        Note: Takes image as a channel first numpy array of shape:
            (batch_size, channels, height, width)
        """
        mat = self._tfshift(matrix, n, 1)
        mat2 = self._tfshift(mat, n, 0)
        return mat2

    def _tf_ifftshift(self, matrix, n):
        """Performs similar function to numpy's ifftshift
        Note: Takes image as a channel first numpy array of shape:
            (batch_size, channels, height, width)
        """
        mat = self._tfshift(matrix, n, 1, invert=True)
        mat2 = self._tfshift(mat, n, 0, invert=True)
        return mat2


class global_average_layer(object):
    def __init__(self, input_x, m=0):
        """
        :param input_x: The input of the last convolution layer, channels last
        """
        with tf.variable_scope('global_average_{0}'.format(m)):
            self.cell_out = tf.reduce_mean(input_x,
                                           axis=(1, 2))

    def output(self):
        return self.cell_out
