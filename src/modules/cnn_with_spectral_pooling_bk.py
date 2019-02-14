"""Implement a CNN with spectral pooling and frequency dropout."""
from .layers import default_conv_layer, spectral_pool_layer
from .layers import spectral_conv_layer
from .layers import fc_layer, global_average_layer
import numpy as np
import tensorflow as tf
import time


class CNN_Spectral_Pool(object):
    """CNN with spectral pooling layers and options for convolution layers."""

    def __init__(self,
                 num_output=10,
                 M=5,
                 conv_filter_size=3,
                 gamma=0.85,
                 alpha=0.3,
                 beta=0.15,
                 weight_decay=1e-3,
                 momentum=0.95,
                 learning_rate=0.0088,
                 l2_norm=0.0001,
                 lr_reduction_epochs=[100, 140],
                 lr_reduction_factor=0.1,
                 max_num_filters=288,
                 random_seed=0,
                 verbose=False,
                 use_spectral_parameterization=False):
        """Initialize model.

        Defaults are set as per the optimum hyperparameters stated in
        the journal article.

        params:
        M = total number of (convolution + spectral-pool) layer-pairs
        """
        self.num_output = num_output
        self.M = M
        self.conv_filter_size = conv_filter_size
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.learning_rate = learning_rate
        self.l2_norm = l2_norm
        self.lr_reduction_epochs = lr_reduction_epochs
        self.lr_reduction_factor = lr_reduction_factor
        self.max_num_filters = max_num_filters
        self.random_seed = random_seed
        self.verbose = verbose
        self.use_spectral_parameterization = use_spectral_parameterization

        # some internal variables:
        self.layers = []

    def _get_cnn_num_filters(self, m):
        """Get number of filters for CNN.

        Args:
            m: current layer number
        """
        return min(self.max_num_filters,
                   96 + 32 * m)

    def _get_sp_dim(self, n):
        """Get filter size for current layer.

        Args:
            n: size of image in layer
        """
        fsize = int(self.gamma * n)
        # minimum size is 3:
        return max(3, fsize)

    def _get_frq_dropout_bounds(self, n, m):
        """Get the bounds for frequency dropout.

        Args:
            n: size of image in layer
            m: current layer index

        Returns:
            freq_dropout_lower_bound: The lower bound for the cutoff
            freq_dropout_upper_bound: The upper bound for the cutoff

        This function implements the linear parameterization of the
        probability distribution for frequency dropout as described in
        section 5.1.
        """
        c = self.alpha + (m / self.M) * (self.beta - self.alpha)

        freq_dropout_lower_bound = c * (1. + n // 2)
        freq_dropout_upper_bound = (1. + n // 2)

        return freq_dropout_lower_bound, freq_dropout_upper_bound

    def _print_message(self, name, args=None):
        """Log a message during graph construction."""
        if not self.verbose:
            return
        if name == 'conv':
            format_str = 'Adding conv layer {0} | Input size: {1} | ' + \
                         'Input channels: {2} | #filters: {3} | ' + \
                         'filter size: {4}'
            print(format_str.format(
                args[0], args[1], args[2], args[3], args[4]
            ))
        if name == 'sp':
            format_str = 'Adding spectral pool layer {0} | ' + \
                         'Input size: {1} | ' + \
                         'filter size: ({2},{2}) | ' + \
                         'Freq Dropout Bounds: ({3},{4})'
            print(format_str.format(
                args[0], args[1], args[2], args[3], args[4]
            ))
        if name == 'softmax':
            print('Adding final softmax layer using global averaging')

        if name == 'final_fc':
            print('Adding final softmax layer using fully-connected layer')

        if name == 'lr_anneal':
            format_str = '\tLearning rate reduced to {0:.4e} at epoch {1}'
            print(format_str.format(self._learning_rate, args))

    def build_graph(
        self, input_x, input_y, train_phase,
        extra_conv_layer=True,
        use_global_averaging=True,
    ):
        """Construct the CNN for training or testing."""
        print("Building tf graph...")

        # variable alias:
        layers = self.layers
        seed = self.random_seed

        # conv layer weights:
        self.conv_layer_weights = []

        # The first part of the graph consists of alternating pairs of
        # convolutional and spectral pooling layers.
        # iterate and define M layers:
        for m in range(1, self.M + 1):
            if m == 1:
                in_x = input_x
            else:
                in_x = layers[-1].output()

            # First add the convolutional layer.
            # get number of channels & image size
            # Note: we're working in channel first domain
            _, nchannel, img_size, _ = in_x.get_shape().as_list()
            nfilters = self._get_cnn_num_filters(m)
            self._print_message(
                'conv',
                (m, img_size, nchannel, nfilters, self.conv_filter_size)
            )
            if self.use_spectral_parameterization:
                conv_layer = spectral_conv_layer(
                    input_x=in_x,
                    in_channel=nchannel,
                    out_channel=nfilters,
                    kernel_size=self.conv_filter_size,
                    random_seed=seed,
                    m=m,
                    data_format='NCHW'
                )
            else:
                conv_layer = default_conv_layer(
                    input_x=in_x,
                    in_channel=nchannel,
                    out_channel=nfilters,
                    kernel_shape=self.conv_filter_size,
                    rand_seed=seed,
                    m=m
                )

            layers.append(conv_layer)
            self.conv_layer_weights.append(conv_layer.weight)

            # Then add the spectral pooling layer
            in_x = conv_layer.output()
            _, _, img_size, _ = in_x.get_shape().as_list()
            filter_size = self._get_sp_dim(img_size)
            freq_dropout_lower_bound, freq_dropout_upper_bound = \
                self._get_frq_dropout_bounds(filter_size, m)
            self._print_message('sp', (
                m,
                img_size,
                filter_size,
                freq_dropout_lower_bound,
                freq_dropout_upper_bound
            ))
            sp_layer = spectral_pool_layer(
                input_x=in_x,
                filter_size=filter_size,
                freq_dropout_lower_bound=freq_dropout_lower_bound,
                freq_dropout_upper_bound=freq_dropout_upper_bound,
                m=m,
                train_phase=train_phase
            )
            layers.append(sp_layer)

        # The alternating pairs of convolutional and spectral pooling layers
        # are followed by a 1x1 convolution
        if extra_conv_layer:
            in_x = layers[-1].output()
            _, nchannel, img_size, _ = in_x.get_shape().as_list()
            nfilters = self._get_cnn_num_filters(self.M)
            self._print_message(
                'conv',
                (self.M + 1, img_size, nchannel, nfilters, 1)
            )
            if self.use_spectral_parameterization:
                layer = spectral_conv_layer(input_x=in_x,
                                            in_channel=nchannel,
                                            out_channel=nfilters,
                                            kernel_size=1,
                                            random_seed=seed,
                                            m=self.M + 1,
                                            data_format='NCHW')

            else:
                layer = default_conv_layer(input_x=in_x,
                                           in_channel=nchannel,
                                           out_channel=nfilters,
                                           kernel_shape=1,
                                           rand_seed=seed,
                                           activation=tf.nn.relu,
                                           m=self.M + 1)
            layers.append(layer)

        # Finally, if we are using global averaging,
        # the last 1x1 convolutional layer is followed by an additional
        # 1x1 convolutional layer with output_dim equal to the possible
        # number of output classes, followed by a final global averaging layer.
        if use_global_averaging:
            # Add last conv layer with same filters as number of classes:
            in_x = layers[-1].output()
            _, nchannel, img_size, _ = in_x.get_shape().as_list()
            nfilters = self.num_output
            self._print_message(
                'conv',
                (self.M + 2, img_size, nchannel, nfilters, 1)
            )
            layer = default_conv_layer(input_x=in_x,
                                       in_channel=nchannel,
                                       out_channel=nfilters,
                                       kernel_shape=1,
                                       rand_seed=seed,
                                       activation=None,
                                       m=self.M + 2)
            layers.append(layer)

            self._print_message('softmax')
            global_average_0 = global_average_layer(layers[-1].output(),
                                                    m=0)
            layers.append(global_average_0)
        else:
            # Alternately, the last convolutional layer can be followed by
            # a fully connected layer.
            self._print_message('final_fc')
            layer = layers[-1]
            pool_shape = layer.output().get_shape()
            img_vector_length = pool_shape[1].value * \
                pool_shape[2].value * \
                pool_shape[3].value
            flatten = tf.reshape(
                layer.output(),
                shape=[-1, img_vector_length]
            )
            fc_layer_0 = fc_layer(
                input_x=flatten,
                in_size=img_vector_length,
                out_size=self.num_output,
                rand_seed=seed,
                activation_function=None
            )
            layers.append(fc_layer_0)

        # update class variables:
        self.layers = layers

        # define loss:
        with tf.name_scope("loss"):
            # l2_loss = tf.reduce_sum([tf.norm(w) for w in fc_w])
            l2_loss = tf.reduce_sum([tf.norm(w, axis=[-2, -1])
                                     for w in self.conv_layer_weights])

            label = tf.one_hot(input_y, self.num_output)
            cross_entropy_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                                                labels=label,
                                                # logits=global_average_0.output()),
                                                logits=layers[-1].output()),
                name='cross_entropy')
            loss = tf.add(cross_entropy_loss,
                          self.l2_norm * l2_loss,
                          name='loss')
            tf.summary.scalar('SP_loss', loss)
        # return global_average_0.output(), loss
        return layers[-1].output(), loss

    def train_step(self, loss, lr):
        """Run one step of the optimizer."""
        with tf.name_scope('train_step'):
            step = tf.train.AdamOptimizer(lr).minimize(loss)

        return step

    def evaluate(self, output, input_y):
        """Calculate the number of errors on the minibatch."""
        with tf.name_scope('evaluate'):
            pred = tf.argmax(output, axis=1)
            error_num = tf.count_nonzero(pred - input_y, name='error_num')
            tf.summary.scalar('model_error_num', error_num)
        return error_num

    def train(self, X_train, y_train, X_val, y_val,
              batch_size=512, epochs=10, val_test_frq=1,
              extra_conv_layer=True,
              use_global_averaging=True,
              model_name='test',
              restore_checkpoint=None):
        """Train the CNN.

        This function was adapted from the homework assignments.
        """
        full_model_name = '{0}_{1}'.format(model_name, time.time())
        self.full_model_name = full_model_name
        self.train_loss = []
        self.val_loss = []
        self.train_accuracy = []
        self.val_accuracy = []

        # defining a copy of learning rate to anneal
        # if by 10% on specified epochs:
        self._learning_rate = self.learning_rate

        # Define the tensorflow variables
        with tf.name_scope('inputs'):
            xs = tf.placeholder(shape=[None, 3, 32, 32], dtype=tf.float32)
            ys = tf.placeholder(shape=[None, ], dtype=tf.int64)
            lr = tf.placeholder(shape=[], dtype=tf.float32)
            train_phase = tf.placeholder(shape=(), dtype=tf.bool)

        output, loss = self.build_graph(
            xs,
            ys,
            train_phase,
            extra_conv_layer,
            use_global_averaging,
        )

        # Calculate the number of minibatches / iterations required
        iters = int(X_train.shape[0] / batch_size)
        val_iters = int(X_val.shape[0] / batch_size)
        print('number of batches for training: {} validation: {}'.format(
            iters,
            val_iters
        ))

        # Define the tensorflow operations needed for updates and
        # error calculations
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            step = self.train_step(loss, lr)
        eve = self.evaluate(output, ys)
        init = tf.global_variables_initializer()

        # Run the training operation.
        with tf.Session() as sess:
            merge = tf.summary.merge_all()
            writer = tf.summary.FileWriter("log/{}/{}".format(
                model_name,
                full_model_name
            ), sess.graph)
            saver = tf.train.Saver()

            sess.run(init)

            if restore_checkpoint is not None:
                print("Restarting training from checkpoint")
                saver.restore(sess, 'model/{}'.format(restore_checkpoint))

            iter_total = 0
            best_acc = 0
            for epc in range(epochs):
                print("training epoch {} ".format(epc + 1))

                # anneal learning rate:
                if (epc + 1) in self.lr_reduction_epochs:
                    self._learning_rate *= self.lr_reduction_factor
                    self._print_message('lr_anneal', epc + 1)

                loss_in_epoch, train_eve_in_epoch = [], []
                for itr in range(iters):
                    iter_total += 1
                    training_batch_x = X_train[itr * batch_size:
                                               (1 + itr) * batch_size]
                    training_batch_y = y_train[itr * batch_size:
                                               (1 + itr) * batch_size]
                    _, iter_loss, iter_eve = sess.run(
                                        [step, loss, eve],
                                        feed_dict={xs: training_batch_x,
                                                   ys: training_batch_y,
                                                   lr: self._learning_rate,
                                                   train_phase: True})
                    loss_in_epoch.append(iter_loss)
                    train_eve_in_epoch.append(iter_eve)

                # Save statistics for the epoch.
                train_loss = np.mean(loss_in_epoch)
                train_eve = np.mean(train_eve_in_epoch)
                train_acc = 100 - train_eve * 100 / training_batch_y.shape[0]
                self.train_loss.append(train_loss)
                self.train_accuracy.append(train_acc)

                # check validation after certain number of
                # epochs as specified in input
                if (epc + 1) % val_test_frq == 0:
                    # do validation
                    val_eves, val_losses, merge_results = [], [], []
                    for val_itr in range(val_iters):
                        val_batch_x = X_val[val_itr * batch_size:
                                            (1 + val_itr) * batch_size]
                        val_batch_y = y_val[val_itr * batch_size:
                                            (1 + val_itr) * batch_size]
                        valid_eve_iter, valid_loss_iter, merge_result_iter = \
                            sess.run(
                                [eve, loss, merge],
                                feed_dict={
                                    xs: val_batch_x,
                                    ys: val_batch_y,
                                    train_phase: False
                                }
                            )
                        val_eves.append(valid_eve_iter)
                        val_losses.append(valid_loss_iter)
                        merge_results.append(merge_result_iter)
                    valid_eve = np.mean(val_eves)
                    valid_loss = np.mean(val_losses)
                    merge_result = merge_results[-1]

                    valid_acc = 100 - valid_eve * 100 / val_batch_y.shape[0]
                    self.val_accuracy.append(valid_acc)
                    self.val_loss.append(valid_loss)
                    if self.verbose:
                        format_str = '{}/{} loss: {} | ' + \
                                     'training accuracy: {:.3f}% | ' + \
                                     'validation accuracy : {:.3f}%'
                        print(format_str.format(
                            batch_size * (itr + 1),
                            X_train.shape[0],
                            train_loss,
                            train_acc,
                            valid_acc))

                    # save the merge result summary
                    writer.add_summary(merge_result, iter_total)

                    # when achieve the best validation accuracy,
                    # we store the model paramters
                    if valid_acc > best_acc:
                        format_str = '\n\tBest validation accuracy! ' + \
                                     'iteration:{} accuracy: {}%\n'
                        print(format_str.format(iter_total, valid_acc))
                        best_acc = valid_acc
                        self.best_acc = best_acc
                        saver.save(sess, 'model/{}/{}'.format(
                            model_name,
                            full_model_name
                        ))

        print("Best validation accuracy: {:.3f}%; Model name: '{}/{}'.".format(
            best_acc,
            model_name,
            full_model_name
        ))

    def calc_test_accuracy(
        self,
        xtest,
        ytest,
        full_model_name,
        batch_size=500
    ):
        """Calculate accuracy for a test set."""
        # restore the last saved best model on this name:
        tf.reset_default_graph()
        with tf.name_scope('inputs'):
            xs = tf.placeholder(shape=[None, 3, 32, 32], dtype=tf.float32)
            ys = tf.placeholder(shape=[None, ], dtype=tf.int64)
            train_phase = tf.placeholder(shape=(), dtype=tf.bool)

        output, _ = self.build_graph(xs, ys, train_phase)
        eve = self.evaluate(output, ys)
        iters = int(xtest.shape[0] / batch_size)
        print('number of batches for testing: {}'.format(
            iters
        ))

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            saver = tf.train.Saver()
            sess.run(init)
            # restore the pre_trained
            print("Loading pre-trained model")
            saver.restore(sess, 'model/{}'.format(full_model_name))
            test_eves = []
            for itr in range(iters):
                X_batch = xtest[itr * batch_size:
                                (1 + itr) * batch_size]
                y_batch = ytest[itr * batch_size:
                                (1 + itr) * batch_size]
                test_iter_eve = sess.run(eve, feed_dict={
                    xs: X_batch,
                    ys: y_batch,
                    train_phase: False
                })
                test_eves.append(test_iter_eve)
            test_eve = np.mean(test_eves)
            test_acc = 100 - test_eve * 100 / y_batch.shape[0]
            print('Test accuracy: {:.3f}'.format(test_acc))
