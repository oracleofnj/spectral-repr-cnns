from .layers import default_conv_layer, spectral_pool_layer, fc_layer
import numpy as np
import tensorflow as tf


class CNN_Spectral_Pool(object):
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
                 l2_norm=0.01,
                 lr_reduction_epochs=[100, 140],
                 lr_reduction_factor=0.1,
                 max_num_filters=288,
                 random_seed=0,
                 verbose=False):
        """Initialize model, defaults are set as per the optimum
        hyperparameters stated in the journal.
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
        self.verbose=verbose

        # some internal variables:
        self.layers = []

    def _get_cnn_num_filters(self, m):
        """ Get number of filters for CNN
        Args:
            m: current layer number
        """
        return min(self.max_num_filters,
                   96 + 32 * m)

    def _get_sp_dim(self, n):
        """Get filter size for current layer
        Args:
            n: size of image in layer
        """
        fsize = int(self.gamma * n)
        # make odd if even
        if fsize % 2 == 0:
            fsize -= 1
        # minimum size is 1:
        return max(1, fsize)

    def _get_frq_dropout(self, n, m):
        """Get the number of dimensions to make 0 in the
        frequency domain.
        Args:
            n: current size of spectral filter
            m: current layer index
        """
        c = self.alpha + (m / self.M) * (self.beta - self.alpha)
        ll = int(c * n)
        ul = m + 1
        ndrop = np.random.uniform(ll, ul)
        return ndrop
    
    def _print_message(self, name, args=None):
        if not self.verbose:
            return
        if name == 'conv':
            print('Adding conv layer {0} | Input size: {1} | Input channels: {2} | #filters: {3} | filter size: {4}'.format(
                                        args[0], args[1], args[2], args[3], args[4]))
        if name == 'sp':
            print('Adding spectral pool layer {0} | Input size: {1} | filter size: ({2},{2})'.format(
                                        args[0], args[1], args[2]))
        if name == 'softmax':
            print('Adding final softmax layer')
            

    def build_graph(self, input_x, input_y):
        print("Building tf graph...")

        # variable alias:
        layers = self.layers
        seed = self.random_seed

        # conv layer weights:
        self.conv_layer_weights = []

        # iterate and define M layers:
        for m in range(1, self.M + 1):
            if m == 1:
                in_x = input_x
            else:
                in_x = layers[-1].output()

            # get number of channels & image size
            # Note: we're working in channel first domain
            _, _, img_size, nchannel = in_x.get_shape().as_list()
            nfilters = self._get_cnn_num_filters(m)
            self._print_message('conv', (m, img_size, nchannel, nfilters, self.conv_filter_size))
            conv_layer = default_conv_layer(input_x=in_x,
                                            in_channel=nchannel,
                                            out_channel=nfilters,
                                            kernel_shape=self.conv_filter_size,
                                            rand_seed=seed,
                                            m=m)
            layers.append(conv_layer)
            self.conv_layer_weights.append(conv_layer.weight)

            # TODO: implement frequency dropout
            in_x = conv_layer.output()
            _, _, img_size, _ = in_x.get_shape().as_list()
            filter_size = self._get_sp_dim(img_size)
            self._print_message('sp', (self.M + 1, img_size, filter_size))
            sp_layer = spectral_pool_layer(input_x=in_x,
                                           filter_size=filter_size)
            layers.append(sp_layer)

        # Add another conv layer:
        in_x = layers[-1].output()
        _, _, img_size, nchannel = in_x.get_shape().as_list()
        nfilters = self._get_cnn_num_filters(self.M)
        self._print_message('conv', (self.M + 1, img_size, nchannel, nfilters, 1))
        layer = default_conv_layer(input_x=in_x,
                                   in_channel=nchannel,
                                   out_channel=nfilters,
                                   kernel_shape=1,
                                   rand_seed=seed,
                                   m=self.M + 1)
        layers.append(layer)

        # Add last conv layer:
        in_x = layers[-1].output()
        _, _, img_size, nchannel = in_x.get_shape().as_list()
        nfilters = 10
        self._print_message('conv', (self.M + 2, img_size, nchannel, nfilters, 1))
        layer = default_conv_layer(input_x=in_x,
                                   in_channel=nchannel,
                                   out_channel=nfilters,
                                   kernel_shape=1,
                                   rand_seed=seed,
                                   m=self.M + 2)
        layers.append(layer)

        # update class variables:
        self.layers = layers

        # final softmax layer:
        # flatten
        self._print_message('softmax')
        pool_shape = layers[-1].output().get_shape()
        img_vector_length = (pool_shape[1].value * pool_shape[2].value *
                             pool_shape[3].value)
        flatten = tf.reshape(layers[-1].output(),
                             shape=[-1, img_vector_length])

        # fc layer
        fc_layer0 = fc_layer(
                            input_x=flatten,
                            in_size=img_vector_length,
                            out_size=self.num_output,
                            rand_seed=seed,
                            activation_function=None,
                            m=0)
        fc_w = [fc_layer0.weight]

        # define loss:
        with tf.name_scope("loss"):
            l2_loss = tf.reduce_sum([tf.norm(w) for w in fc_w])
            l2_loss += tf.reduce_sum([tf.norm(w, axis=[-2, -1])
                                      for w in self.conv_layer_weights])

            label = tf.one_hot(input_y, self.num_output)
            cross_entropy_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                                                labels=label,
                                                logits=fc_layer0.output()),
                name='cross_entropy')
            loss = tf.add(cross_entropy_loss,
                          self.l2_norm * l2_loss,
                          name='loss')
            tf.summary.scalar('SP_loss', loss)
        return fc_layer0.output(), loss

    def train_step(self, loss):
        with tf.name_scope('train_step'):
            step = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

        return step

    def evaluate(self, output, input_y):
        with tf.name_scope('evaluate'):
            pred = tf.argmax(output, axis=1)
            error_num = tf.count_nonzero(pred - input_y, name='error_num')
            tf.summary.scalar('LeNet_error_num', error_num)
        return error_num

    def train(self, X_train, y_train, X_val, y_val,
              batch_size=512, epochs=10, val_test_frq=20,
              model_name='test'):
        self.loss_vals = []
        self.train_accuracy = []
        self.val_accuracy = []
        with tf.name_scope('inputs'):
            xs = tf.placeholder(shape=[None, 32, 32, 3], dtype=tf.float32)
            ys = tf.placeholder(shape=[None, ], dtype=tf.int64)

        output, loss = self.build_graph(xs, ys)
        # print(type(loss))
        iters = int(X_train.shape[0] / batch_size)
        print('number of batches for training: {}'.format(iters))

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            step = self.train_step(loss)
        eve = self.evaluate(output, ys)

        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            merge = tf.summary.merge_all()
            writer = tf.summary.FileWriter("log/{}".format(model_name), sess.graph)
            saver = tf.train.Saver()

            sess.run(init)

            iter_total = 0
            best_acc = 0
            for epc in range(epochs):
                print("training epoch {} ".format(epc + 1))

                for itr in range(iters):
                    iter_total += 1
                    # if self.verbose:
                    #     print("\tTraining batch {0}".format(itr))

                    training_batch_x = X_train[itr * batch_size:
                                               (1 + itr) * batch_size]
                    training_batch_y = y_train[itr * batch_size:
                                               (1 + itr) * batch_size]

                    _, cur_loss, train_eve = sess.run(
                                        [step, loss, eve],
                                        feed_dict={xs: training_batch_x,
                                                   ys: training_batch_y})
                    self.loss_vals.append(cur_loss)
                
                # check validation after certain number of epochs as specified in input
                if (epc + 1) % val_test_frq == 0:
                    # do validation
                    valid_eve, merge_result = sess.run([eve, merge],
                                                       feed_dict={
                                                       xs: X_val,
                                                       ys: y_val})
                    valid_acc = 100 - valid_eve * 100 / y_val.shape[0]
                    train_acc = 100 - train_eve * 100 / training_batch_y.shape[0]
                    self.train_accuracy.append(train_acc)
                    self.val_accuracy.append(valid_acc)
                    if verbose:
                        print('{}/{} loss: {} | training accuracy: {} | validation accuracy : {}%'.format(
                            batch_size * (itr + 1),
                            X_train.shape[0],
                            cur_loss,
                            train_acc,
                            valid_acc))

                    # save the merge result summary
                    writer.add_summary(merge_result, iter_total)

                    # when achieve the best validation accuracy, we store the model paramters
                    if valid_acc > best_acc:
                        print('Best validation accuracy! iteration:{} accuracy: {}%'.format(iter_total, valid_acc))
                        best_acc = valid_acc
                        saver.save(sess, 'model/{}'.format(model_name))

        print("Traning ends. The best valid accuracy is {}. Model named {}.".format(best_acc, model_name))


