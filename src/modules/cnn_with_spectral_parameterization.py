from .layers import spectral_conv_layer, global_average_layer
import numpy as np
import tensorflow as tf
from .image_generator import ImageGenerator

class CNN_Spectral_Param():
	"""
	This class builds and trains the generic and deep CNN architectures 
	as described in section 5.2 of the paper with and without spectral pooling.
	"""
	def __init__(self,
		num_output=10,
		architecture='generic',
		use_spectral_params=True,
		kernel_size=3,
		l2_norm=0.01,
		learning_rate=1e-4,
		data_format='NHWC',
		random_seed=0):
		"""
		:param num_output: Number of classes to predict
		:param arcchitecture: Defines which architecture to build (either deep or generic)
		:param use_spectral_params: Flag to turn spectral parameterization on and off
		:param kernel_size: size of convolutional kernel
		:param l2_norm: Scale factor for l2 norm of CNN weights when calculating l2 loss
		:learning_rate: Learning rate for Adam AdamOptimizer
		:data_format: Format of input images, either 'NHWC' or 'NCHW'
		:random_seed: Seed for initializers to create reproducable results
		"""
		self.num_output = num_output
		self.architecture = architecture
		self.use_spectral_params = use_spectral_params
		self.kernel_size = kernel_size
		self.l2_norm = l2_norm
		self.learning_rate = learning_rate
		self.random_seed = random_seed

	def build_graph(self, input_x, input_y):
		"""
		This function calls one of two helper functions to build the CNN graph

		:param input_x: 4D array containing images to train model on
		:param input_y: 1D array containing class labels of images
		"""
		if self.architecture == 'generic':
			return self._build_generic_architecture(input_x, input_y)
		elif self.architecture == 'deep':
			return self._build_deep_architecture(input_x, input_y)
		else:
			raise Exception('Architecture \'' + self.architecture + '\' not defined')

	def train_step(self, loss):
		"""
		Calls Adam optimizer to minimize inputted loss

		:param loss: the loss to minimize
		"""
		with tf.name_scope('train_step'):
			step = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

		return step

	def evaluate(self, pred, input_y):
		"""
		Calculates the number of errors made in the prediction array and returns them

		:param pred: The prediction array for the class labels
		:param input_y: The ground-truth y values
		"""
		with tf.name_scope('evaluate'):
			error_num = tf.count_nonzero(pred - input_y, name='error_num')
			tf.summary.scalar('LeNet_error_num', error_num)
		return error_num

	def train(self, X_train, y_train, batch_size=512, epochs=10):
		"""
		Trains the CNN model. This is where data augmentation is added and
		the training accuracy is tracked.

		:param X_train: 4D training set (num images, height, width, num channels)
		:param y_train: 1D training labels
		:param batch_size: Number of images to include in the minibatch,
		before applying gradient updates
		:param epochs: Number of epochs to train the model for
		"""

		# Instantiate image generator for data augmentation
		img_gen = ImageGenerator(X_train, y_train)
		img_gen.translate(shift_height=-2, shift_width=0)
		generator = img_gen.next_batch_gen(batch_size)

		# Variables to track different metrics we're interested in
		self.loss_vals = []
		self.train_accuracy = []
		self.error_rate = []
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

					# Apply vertical translations and random horizontal flips
					if epc % 4 == 0 or epc % 4 == 1:
						img_gen.translate(shift_height=2, shift_width=0)
					elif epc % 4 == 2 or epc % 4 == 3:
						img_gen.translate(shift_height=-2, shift_width=0)
					else:
						print('This is bad...')

					if np.random.randint(2, size=1)[0] == 1:
						img_gen.flip(mode='h')

					loss_in_epoch = []
					train_acc_in_epoch = []
					error_rate_in_epoch = []
					for itr in range(iters):
						iter_total += 1

						training_batch_x, training_batch_y = next(generator)

						_, cur_loss, error_num = sess.run(
											[step, loss, eve],
											feed_dict={xs: training_batch_x,
													   ys: training_batch_y})
						loss_in_epoch.append(cur_loss)
						train_acc_in_epoch.append(1 - error_num / batch_size)
						error_rate_in_epoch.append(error_num / batch_size)

					self.loss_vals.append(np.mean(loss_in_epoch))
					self.train_accuracy.append(np.mean(train_acc_in_epoch))
					self.error_rate.append(np.mean(error_rate_in_epoch))

					print('Error rate:',self.error_rate[-1])
					print('Train acc:',self.train_accuracy[-1])
					print('Loss:',self.loss_vals[-1])

	def _build_generic_architecture(self, input_x, input_y):
		"""
		Builds the generic architecture (defined in section 5.2 of the paper)

		This architecture is a pair of convolution and max-pool layers, followed
		by three fully-connected layers and a softmax.

		:param input_x: 4D training set
		:param input_y: 1D training labels
		"""
		spatial_conv_weights = []

		# These if statements decide whether we'll use spectral convolution or 
		# the built-in tensorflow convolutional layer
		if self.use_spectral_params:
			sc_layer = spectral_conv_layer(input_x=input_x,
										in_channel=3,
										out_channel=96,
										kernel_size=self.kernel_size,
										random_seed=self.random_seed,
										m=1)
			conv1 = sc_layer.output()
			spatial_conv_weights.append(sc_layer.weight)

		else:
			conv1 = tf.layers.conv2d(inputs=input_x,
										filters=96,
										kernel_size=self.kernel_size,
										activation=tf.nn.relu,
										padding='SAME',
										name='conv1')

		pool1 = tf.layers.max_pooling2d(inputs=conv1,
										pool_size=3,
										strides=2,
										padding='SAME',
										name='max_pool_1')

		if self.use_spectral_params:
			sc_layer = spectral_conv_layer(input_x=pool1,
										in_channel=96,
										out_channel=192,
										kernel_size=self.kernel_size,
										random_seed=self.random_seed,
										m=2)
			conv2 = sc_layer.output()
			spatial_conv_weights.append(sc_layer.weight)

		else:
			conv2 = tf.layers.conv2d(inputs=pool1,
										filters=192,
										kernel_size=self.kernel_size,
										activation=tf.nn.relu,
										padding='SAME',
										name='conv2')

		pool2 = tf.layers.max_pooling2d(inputs=conv2,
										pool_size=3,
										strides=2,
										padding='SAME',
										name='max_pool_2')

		flatten = tf.contrib.layers.flatten(inputs=pool2)

		fc1 = tf.contrib.layers.fully_connected(inputs=flatten,
												num_outputs=1024,
												activation_fn=tf.nn.relu)
		fc2 = tf.contrib.layers.fully_connected(inputs=fc1,
												num_outputs=512,
												activation_fn=tf.nn.relu)
		fc3 = tf.contrib.layers.fully_connected(inputs=fc2,
												num_outputs=self.num_output,
												activation_fn=None)

		fc_weights = [v for v in tf.trainable_variables() if 'weights' in v.name]

		with tf.name_scope("loss"):
			# Calculating l2 norms for the loss
			if self.use_spectral_params:
				l2_loss = tf.reduce_sum([tf.norm(w, axis=[-2, -1]) for w in spatial_conv_weights])
			else:
				conv_kernels = [v for v in tf.trainable_variables() if 'kernel' in v.name]
				l2_loss = tf.reduce_sum([tf.norm(w, axis=[-2, -1]) for w in conv_kernels])

			l2_loss += tf.reduce_sum([tf.norm(w) for w in fc_weights])

			# Calculating cross entropy loss
			label = tf.one_hot(input_y, self.num_output)
			cross_entropy_loss = tf.reduce_mean(
				tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=fc3),
				name='cross_entropy')
			loss = tf.add(cross_entropy_loss, self.l2_norm * l2_loss, name='loss')

		# Returns output of the final layer as well as the loss
		return fc3, loss

	def _build_deep_architecture(self, input_x, input_y):
		"""
		Builds the deep architecture (defined in section 5.2 of the paper)

		This architecture is defined as follows:
			back-to-back convolutions, max-pool, back-to-back-to-back convolutions,
			max-pool, back-to-back 1-filter convolutions, and a global averaging

		:param input_x: 4D training set
		:param input_y: 1D training labels
		"""
		spatial_conv_weights = []

		# Again, these if-statements determine whether to use spectral conv or default
		if self.use_spectral_params:
			sc_layer = spectral_conv_layer(input_x=input_x,
											in_channel=3,
											out_channel=96,
											kernel_size=self.kernel_size,
											random_seed=self.random_seed,
											m=1)
			conv1 = sc_layer.output()
			spatial_conv_weights.append(sc_layer.weight)
		else:
			conv1 = tf.layers.conv2d(inputs=input_x,
										filters=96,
										kernel_size=self.kernel_size,
										activation=tf.nn.relu,
										padding='SAME',
										name='conv1')

		if self.use_spectral_params:
			sc_layer = spectral_conv_layer(input_x=conv1,
											in_channel=96,
											out_channel=96,
											kernel_size=self.kernel_size,
											random_seed=self.random_seed,
											m=2)
			conv2 = sc_layer.output()
			spatial_conv_weights.append(sc_layer.weight)
		else:
			conv2 = tf.layers.conv2d(inputs=conv1,
										filters=96,
										kernel_size=self.kernel_size,
										activation=tf.nn.relu,
										padding='SAME',
										name='conv2')

		pool1 = tf.layers.max_pooling2d(inputs=conv2,
										pool_size=3,
										strides=2,
										padding='SAME',
										name='max_pool_1')

		if self.use_spectral_params:
			sc_layer = spectral_conv_layer(input_x=pool1,
											in_channel=96,
											out_channel=192,
											kernel_size=self.kernel_size,
											random_seed=self.random_seed,
											m=3)
			conv3 = sc_layer.output()
			spatial_conv_weights.append(sc_layer.weight)
		else:
			conv3 = tf.layers.conv2d(inputs=pool1,
										filters=192,
										kernel_size=self.kernel_size,
										activation=tf.nn.relu,
										padding='SAME',
										name='conv3')

		if self.use_spectral_params:
			sc_layer = spectral_conv_layer(input_x=conv3,
											in_channel=192,
											out_channel=192,
											kernel_size=self.kernel_size,
											random_seed=self.random_seed,
											m=4)
			conv4 = sc_layer.output()
			spatial_conv_weights.append(sc_layer.weight)
		else:
			conv4 = tf.layers.conv2d(inputs=conv3,
										filters=192,
										kernel_size=self.kernel_size,
										activation=tf.nn.relu,
										padding='SAME',
										name='conv4')

		if self.use_spectral_params:
			sc_layer = spectral_conv_layer(input_x=conv4,
											in_channel=192,
											out_channel=192,
											kernel_size=self.kernel_size,
											random_seed=self.random_seed,
											m=5)
			conv5 = sc_layer.output()
			spatial_conv_weights.append(sc_layer.weight)
		else:
			conv5 = tf.layers.conv2d(inputs=conv4,
										filters=192,
										kernel_size=self.kernel_size,
										activation=tf.nn.relu,
										padding='SAME',
										name='conv5')

		pool2 = tf.layers.max_pooling2d(inputs=conv5,
										pool_size=3,
										strides=2,
										padding='SAME',
										name='max_pool_2')

		if self.use_spectral_params:
			sc_layer = spectral_conv_layer(input_x=pool2,
											in_channel=192,
											out_channel=192,
											kernel_size=1,
											random_seed=self.random_seed,
											m=6)
			conv6 = sc_layer.output()
			spatial_conv_weights.append(sc_layer.weight)
		else:
			conv6 = tf.layers.conv2d(inputs=pool2,
										filters=192,
										kernel_size=1,
										activation=tf.nn.relu,
										padding='SAME',
										name='conv6')

		if self.use_spectral_params:
			sc_layer = spectral_conv_layer(input_x=conv6,
											in_channel=192,
											out_channel=10,
											kernel_size=1,
											random_seed=self.random_seed,
											m=7)
			conv7 = sc_layer.output()
			spatial_conv_weights.append(sc_layer.weight)
		else:
			conv7 = tf.layers.conv2d(inputs=conv6,
										filters=10,
										kernel_size=1,
										activation=None,
										padding='SAME',
										name='conv7')

		global_avg = tf.reduce_mean(input_tensor=conv7, axis=[1,2])

		with tf.name_scope("loss"):
			if self.use_spectral_params:
				l2_loss = tf.reduce_sum([tf.norm(w, axis=[-2, -1]) for w in spatial_conv_weights if w.shape[0] == 3])
				l2_loss += tf.reduce_sum([tf.norm(w, axis=[-2, -1]) for w in spatial_conv_weights if w.shape[0] == 1])
			else:
				conv_kernels = [v for v in tf.trainable_variables() if 'kernel' in v.name]
				l2_loss = tf.reduce_sum([tf.norm(w, axis=[-2, -1]) for w in conv_kernels if w.shape[0] == 3])
				l2_loss += tf.reduce_sum([tf.norm(w, axis=[-2, -1]) for w in conv_kernels if w.shape[0] == 1])

			label = tf.one_hot(input_y, self.num_output)
			cross_entropy_loss = tf.reduce_mean(
				tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=global_avg),
				name='cross_entropy')
			loss = tf.add(cross_entropy_loss, self.l2_norm * l2_loss, name='loss')

		return global_avg, loss

