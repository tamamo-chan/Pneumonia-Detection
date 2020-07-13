import math

import numpy as np
import keras
from PIL import Image
from keras.utils import Sequence


class DataGenerator(Sequence):

    def __init__(self, image_paths, labels, dim=(224, 224), batch_size=16):
        self.dim = dim
        self.labels = labels
        self.image_paths = image_paths
        self.batch_size = batch_size

    def __len__(self):
        'Denotes the number of batches per epoch'
        return math.ceil(len(self.labels) / self.batch_size)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        from_index = index * self.batch_size
        to_index = (index + 1) * self.batch_size
        batch_x = self.image_paths[from_index: to_index]
        batch_y = self.labels[from_index: to_index]

        for i in range(len(batch_x)):
            batch_x[i] = self.__augmentor__(batch_x[i])
        return np.array(batch_x), np.array(batch_y)

    def __augmentor__(self, file):
        image = np.loadtxt(file)

        im = Image.fromarray(image).convert('RGB')
        new_img = im.resize(self.dim)

        rgb_array = np.array(new_img)

        return rgb_array
