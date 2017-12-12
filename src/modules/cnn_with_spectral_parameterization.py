from .layers import spectral_conv_layer, fc_layer
import numpy as np
import tensorflow as tf

class CNN_Spectral_Param():
	def __init__(self,
		num_output,
		architecture='generic',
		l2_norm=0.01,
		learning_rate=1e-3,
		random_seed=0):

		self.num_output = num_output
		self.architecture = architecture
		self.l2_norm = l2_norm
		self.learning_rate = learning_rate
		self.random_seed = random_seed

	def build_graph(self, input_x, input_y):
		if self.architecture == 'generic':
			return self._build_generic_architecture(input_x, input_y)
		elif self.architecture == 'deep':
			return self._build_deep_architecture(input_x, input_y)
		else:
			raise Exception('Architecture \'' + self.architecture + '\' not defined')

	def train_step(self, loss):
		with tf.name_scope('train_step'):
			step = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

		return step

	def evaluate(self, pred, input_y):
		with tf.name_scope('evaluate'):
			# pred = tf.argmax(output, axis=1)
			error_num = tf.count_nonzero(pred - input_y, name='error_num')
			tf.summary.scalar('LeNet_error_num', error_num)
		return error_num

	def train(self, X_train, y_train, X_val, y_test,
			  batch_size=512, epochs=10, val_test_frq=20):
		self.loss_vals = []
		self.train_accuracy = []
		with tf.name_scope('inputs'):
			xs = tf.placeholder(shape=[None, 32, 32, 3], dtype=tf.float32)
			ys = tf.placeholder(shape=[None, ], dtype=tf.int64)

			output, loss = self.build_graph(xs, ys)
			iters = int(X_train.shape[0] / batch_size)
			print('number of batches for training: {}'.format(iters))

			update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
			with tf.control_dependencies(update_ops):
				step = self.train_step(loss)
			pred = tf.argmax(output, axis=1)
			eve = self.evaluate(pred, ys)

			init = tf.global_variables_initializer()
			with tf.Session() as sess:
				sess.run(init)

				iter_total = 0
				for epc in range(epochs):
					print("epoch {} ".format(epc + 1))

					for itr in range(iters):
						iter_total += 1
						print("Training batch {0}".format(itr))

						training_batch_x = X_train[itr * batch_size:
												   (1 + itr) * batch_size]
						training_batch_y = y_train[itr * batch_size:
												   (1 + itr) * batch_size]

						_, cur_loss, train_eve = sess.run(
											[step, loss, eve],
											feed_dict={xs: training_batch_x,
													   ys: training_batch_y})
						self.loss_vals.append(cur_loss)
						self.train_accuracy.append(1 - train_eve / batch_size)
					print(self.train_accuracy[-1])

	def _build_generic_architecture(self, input_x, input_y):

		conv1 = spectral_conv_layer(input_x=input_x,
									in_channel=input_x.shape[-1],
									out_channel=96,
									kernel_shape=3,
									random_seed=self.random_seed,
									m=1)

		pool1_output = tf.layers.max_pooling2d(inputs=conv1.output(),
										pool_size=3,
										strides=2,
										padding='SAME',
										name='max_pool_1')

		conv2 = spectral_conv_layer(input_x=pool1_output,
									in_channel=pool1_output.shape[-1],
									out_channel=192,
									kernel_shape=3,
									random_seed=self.random_seed,
									m=2)

		pool2_output = tf.layers.max_pooling2d(inputs=conv2.output(),
										pool_size=3,
										strides=2,
										padding='SAME',
										name='max_pool_2')

		pool_shape = pool2_output.get_shape()
		img_vector_length = pool_shape[1].value * pool_shape[2].value * pool_shape[3].value
		flatten = tf.reshape(pool2_output, shape=[-1, img_vector_length])

		fc1 = fc_layer(input_x=flatten,
						in_size=img_vector_length,
						out_size=1024,
						rand_seed=self.random_seed,
						activation_function=tf.nn.relu,
						m=1)

		fc2 = fc_layer(input_x=fc1.output(),
						in_size=1024,
						out_size=512,
						rand_seed=self.random_seed,
						activation_function=tf.nn.relu,
						m=2)

		fc3 = fc_layer(input_x=fc2.output(),
						in_size=512,
						out_size=self.num_output,
						rand_seed=self.random_seed,
						activation_function=None,
						m=3)

		output = tf.nn.softmax(fc3.output())

		with tf.name_scope("loss"):
			# l2_loss = tf.reduce_sum([tf.norm(w) for w in [fc1.weight, fc2.weight, fc3.weight]])
			# l2_loss += tf.reduce_sum([tf.norm(w, axis=[-2, -1]) for w in [conv1.weight, conv2.weight]])

			label = tf.one_hot(input_y, self.num_output)
			cross_entropy_loss = tf.reduce_mean(
				tf.nn.softmax_cross_entropy_with_logits(labels=label,logits=output),
				name='cross_entropy')
			loss = cross_entropy_loss
			# loss = tf.add(cross_entropy_loss,self.l2_norm * l2_loss,name='loss')

		return output, loss

	def _build_deep_architecture(self, input_x, input_y):
		pass
