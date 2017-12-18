"""Implements layers for the spectral CNN."""
from .spectral_pool import _common_spectral_pool
from .frequency_dropout import _frequency_dropout_mask
import numpy as np
import tensorflow as tf


class default_conv_layer(object):
    def __init__(self, input_x, in_channel, out_channel,
                 kernel_shape, rand_seed,
                 activation=tf.nn.relu,
                 m=0):
        """
        NOTE: Image should be CHANNEL FIRST
        This class has been partially adapted from the homework assignments
        given by the TAs
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
            if activation is not None:
                cell_out = activation(cell_biased)
            else:
                cell_out = cell_biased

            self.cell_out = cell_out

            tf.summary.histogram(
                'conv_layer/{}/kernel'.format(m),
                weight
            )
            tf.summary.histogram(
                'conv_layer/{}/bias'.format(m),
                bias
            )
            tf.summary.histogram(
                'conv_layer/{}/activation'.format(m),
                cell_out
            )

    def output(self):
        return self.cell_out


class fc_layer(object):
    def __init__(self, input_x, in_size, out_size, rand_seed,
                 activation_function=None, m=0):
        """
        Implementing fully-connected layers.
        This class has been partially adapted from the homework assignments
        given by the TAs
        :param input_x: The input of the FC layer. It should be a
            flatten vector.
        :param in_size: The length of input vector.
        :param out_size: The length of output vector.
        :param rand_seed: An integer that presents the random seed used
            to generate the initial parameter value.
        :param keep_prob: The probability of dropout. Default set by
            1.0 (no drop-out applied)
        :param activation_function: The activation function
            for the output. Default set to None.
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
        """Return layer output."""
        return self.cell_out


class spectral_pool_layer(object):
    """Spectral pooling layer."""

    def __init__(
        self,
        input_x,
        filter_size=3,
        freq_dropout_lower_bound=None,
        freq_dropout_upper_bound=None,
        activation=tf.nn.relu,
        m=0,
        train_phase=False
    ):
        """Perform a single spectral pool operation.

        Args:
            input_x: Tensor representing a batch of channels-first images
                shape: (batch_size, num_channels, height, width)
            filter_size: int, the final dimension of the filter required
            freq_dropout_lower_bound: The lowest possible frequency
                above which all frequencies should be truncated
            freq_dropout_upper_bound: The highest possible frequency
                above which all frequencies should be truncated
            train_phase: tf.bool placeholder or Python boolean,
                but using a Python boolean is probably wrong

        Returns:
            An image of similar shape as input after reduction
        """
        # assert only 1 dimension passed for filter size
        assert isinstance(filter_size, int)

        input_shape = input_x.get_shape().as_list()
        assert len(input_shape) == 4
        _, _, H, W = input_shape
        assert H == W

        with tf.variable_scope('spectral_pool_layer_{0}'.format(m)):
            # Compute the Fourier transform of the image
            im_fft = tf.fft2d(tf.cast(input_x, tf.complex64))

            # Truncate the spectrum
            im_transformed = _common_spectral_pool(im_fft, filter_size)
            if (
                freq_dropout_lower_bound is not None and
                freq_dropout_upper_bound is not None
            ):
                # If we are in the training phase, we need to drop all
                # frequencies above a certain randomly determined level.
                def true_fn():
                    tf_random_cutoff = tf.random_uniform(
                        [],
                        freq_dropout_lower_bound,
                        freq_dropout_upper_bound
                    )
                    dropout_mask = _frequency_dropout_mask(
                        filter_size,
                        tf_random_cutoff
                    )
                    return im_transformed * dropout_mask

                # In the testing phase, return the truncated frequency
                # matrix unchanged.
                def false_fn():
                    return im_transformed

                im_downsampled = tf.cond(
                    train_phase,
                    true_fn=true_fn,
                    false_fn=false_fn
                )
                im_out = tf.real(tf.ifft2d(im_downsampled))
            else:
                im_out = tf.real(tf.ifft2d(im_transformed))

            if activation is not None:
                cell_out = activation(im_out)
            else:
                cell_out = im_out
            tf.summary.histogram('sp_layer/{}/activation'.format(m), cell_out)

        self.cell_out = cell_out

    def output(self):
        return self.cell_out


class spectral_conv_layer(object):
    def __init__(self, input_x, in_channel, out_channel,
                 kernel_size, random_seed, data_format='NHWC', m=0):
        """
        NOTE: Image should be CHANNEL LAST
        :param input_x: Should be a 4D array like:
                            (batch_num, channel_num, img_len, img_len)
        :param in_channel: The number of channels
        :param out_channel: number of filters required
        :param kernel_size: kernel size
        :param random_seed: random seed
        :param index: The layer index used for naming
        """
        assert len(input_x.shape) == 4
        if data_format == 'NHWC':
            assert input_x.shape[1] == input_x.shape[2]
            assert input_x.shape[3] == in_channel
        elif data_format == 'NCHW':
            assert input_x.shape[1] == in_channel
            assert input_x.shape[2] == input_x.shape[3]


        def _glorot_sample(kernel_size, n_in, n_out):
            limit = np.sqrt(6 / (n_in + n_out))
            return np.random.uniform(
                low=-limit,
                high=limit,
                size=(n_in, n_out, kernel_size, kernel_size)
            )

        with tf.variable_scope('spec_conv_layer_{0}'.format(m)):
            with tf.name_scope('spec_conv_kernel'):
                samp = _glorot_sample(kernel_size, in_channel, out_channel)
                spectral_weight_init = tf.transpose(
                    tf.fft2d(samp),
                    [2, 3, 0, 1]
                )

                real_init = tf.get_variable(
                    name='real_{0}'.format(m),
                    initializer=tf.real(spectral_weight_init))

                imag_init = tf.get_variable(
                    name='imag_{0}'.format(m),
                    initializer=tf.imag(spectral_weight_init))

                spectral_weight = tf.complex(
                    real_init,
                    imag_init,
                    name='spectral_weight_{0}'.format(m)
                )
                self.spectral_weight = spectral_weight

            with tf.variable_scope('conv_bias'):
                b_shape = [out_channel]
                bias = tf.get_variable(
                    name='conv_bias_{0}'.format(m),
                    shape=b_shape,
                    initializer=tf.glorot_uniform_initializer(
                        seed=random_seed
                    ))
                self.bias = bias

            complex_spatial_weight = tf.transpose(tf.ifft2d(tf.transpose(
                spectral_weight, [2, 3, 0, 1])),
                [2, 3, 0, 1]
            )
            spatial_weight = tf.real(
                complex_spatial_weight,
                name='spatial_weight_{0}'.format(m)
            )
            self.weight = spatial_weight

            conv_out = tf.nn.conv2d(input_x, spatial_weight,
                                    strides=[1, 1, 1, 1],
                                    padding="SAME",
                                    data_format=data_format)
            self.cell_out = tf.nn.relu(tf.nn.bias_add(conv_out, bias, data_format=data_format))

    def output(self):
        return self.cell_out


class global_average_layer(object):
    def __init__(self, input_x, m=0):
        """
        :param input_x: The input of the last convolution layer, channels last
        """
        with tf.variable_scope('global_average_{0}'.format(m)):
            self.cell_out = tf.reduce_mean(input_x,
                                           axis=(2, 3))
            print(self.cell_out.get_shape())

    def output(self):
        return self.cell_out
