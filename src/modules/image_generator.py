import numpy as np
from matplotlib import pyplot as plt

class ImageGenerator(object):

    def __init__(self, x, y):
        """
        Initialize an ImageGenerator instance.
        :param x: A Numpy array of input data. It has shape (num_of_samples, height, width, channels).
        :param y: A Numpy vector of labels. It has shape (num_of_samples, ).
        """
        self.x = x.copy()
        self.y = y
        self.num_samples, self.height, self.width, self.num_channels = self.x.shape
        self.shift_height = 0
        self.shift_width = 0
        self.angle = 0
        self.is_horizontal_flip = False
        self.is_vertical_flip = False
        self.is_add_noise = False

    def next_batch_gen(self, batch_size, shuffle=True):
        """
        A python generator function that yields a batch of data indefinitely.
        :param batch_size: The number of samples to return for each batch.
        :param shuffle: If True, shuffle the entire dataset after every sample has been returned once.
                        If False, the order or data samples stays the same.
        :return: A batch of data with size (batch_size, width, height, channels).
        """
        total_batches = self.num_samples // batch_size
        batch_count = 0
        while True:
            if batch_count < total_batches:
                batch_count += 1
                yield (self.x[(batch_count-1) * batch_size : batch_count * batch_size], 
                       self.y[(batch_count-1) * batch_size : batch_count * batch_size])
            else:
                if shuffle:
                    perm = np.random.permutation(self.num_samples)
                    self.x = self.x[perm]
                    self.y = self.y[perm]
                batch_count = 0

    def show(self):
        """
        Plot the top 16 images (index 0~15) of self.x for visualization.
        """
        X_sample = self.x[:16]

        # Visualize one channel of images 
        r = 4
        f, axarr = plt.subplots(r, r, figsize=(8,8))
        for i in range(r):
            for j in range(r):
                img = X_sample[r*i+j]
                axarr[i][j].imshow(img, cmap="gray")

    def translate(self, shift_height, shift_width):
        """
        Translate self.x by the values given in shift.
        :param shift_height: the number of pixels to shift along height direction. Can be negative.
        :param shift_width: the number of pixels to shift along width direction. Can be negative.
        :return:
        """
        self.shift_height = shift_height
        self.shift_width = shift_width
        
        self.x = np.roll(self.x, shift_height, axis=1)
        self.x = np.roll(self.x, shift_width, axis=2)

    def flip(self, mode='h'):
        """
        Flip self.x according to the mode specified
        :param mode: 'h' or 'v' or 'hv'. 'h' means horizontal and 'v' means vertical.
        """
        self.is_horizontal_flip = 'h' in mode
        self.is_vertical_flip = 'v' in mode
        if self.is_horizontal_flip:
            self.x = np.flip(self.x, axis=2)
        if self.is_vertical_flip:
            self.x = np.flip(self.x, axis=1)