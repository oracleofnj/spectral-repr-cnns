from .cnn_with_spectral_pooling import CNN_Spectral_Pool
import tensorflow as tf
import numpy as np


def train_step(loss, learning_rate=1e-3):
    with tf.name_scope('train_step'):
        step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    return step


def evaluate(pred, input_y):
    with tf.name_scope('evaluate'):
        # pred = tf.argmax(output, axis=1)
        error_num = tf.count_nonzero(pred - input_y, name='error_num')
        tf.summary.scalar('LeNet_error_num', error_num)
    return error_num


def train(X_train, y_train,
          X_val=None, y_val=None,
          batch_size=512,
          architechture='cnn_spectral_pool',
          arch_params={}):
    # define the variables and parameter needed during training
    with tf.name_scope('inputs'):
        xs = tf.placeholder(shape=[None, 3, 32, 32], dtype=tf.float32)
        ys = tf.placeholder(shape=[None, ], dtype=tf.int64)
        # train_phase = tf.placeholder(shape=(), dtype=tf.bool)

    if architechture == 'cnn_spectral_pool':
        arch = CNN_Spectral_Pool(
                    X_train=xs, y_train=ys,
                    **arch_params)
    output, loss = arch.build_graph()

    iters = int(X_train.shape[0] / batch_size)
    print('number of batches for training: {}'.format(iters))

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        step = train_step(loss)
    eve = evaluate(output, ys)

    iter_total = 0
    best_acc = 0

    if model_name is None:
        cur_model_name = 'lenet_{}'.format(int(time.time()))
    else:
        cur_model_name = 'lenet_{}'.format(model_name)

    with tf.Session() as sess:
        merge = tf.summary.merge_all()

        writer = tf.summary.FileWriter("log/{}".format(cur_model_name), sess.graph)
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        # try to restore the pre_trained
        if pre_trained_model is not None:
            try:
                print("Load the model from: {}".format(pre_trained_model))
                saver.restore(sess, 'model/{}'.format(pre_trained_model))
            except Exception:
                print("Load model Failed!")
                pass

        for epc in range(epoch):
            print("epoch {} ".format(epc + 1))

            for itr in range(iters):
                iter_total += 1

                training_batch_x = X_train[itr * batch_size: (1 + itr) * batch_size]
                training_batch_y = y_train[itr * batch_size: (1 + itr) * batch_size]

                _, cur_loss, train_eve = sess.run([step, loss, eve],
                                                  feed_dict={
                                       xs: training_batch_x,
                                       ys: training_batch_y,
                                       train_phase: True})

                if iter_total % 100 == 0:
                    # do validation
                    valid_eve, merge_result = sess.run([eve, merge],
                                                       feed_dict={
                                                       xs: X_val,
                                                       ys: y_val,
                                                       train_phase: False})
                    valid_acc = 100 - valid_eve * 100 / y_val.shape[0]
                    train_acc = 100 - train_eve * 100 / training_batch_y.shape[0]
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
                        saver.save(sess, 'model/{}'.format(cur_model_name))

    print("Traning ends. The best valid accuracy is {}. Model named {}.".format(best_acc, cur_model_name))