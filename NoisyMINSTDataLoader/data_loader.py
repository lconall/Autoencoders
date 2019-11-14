import numpy as np
import random
from keras.datasets import mnist

import KerasAutoencoder
from KerasAutoencoder.interfaces.data_loader_interface import DataLoaderInterface


class NoisyMINSTDataLoader(DataLoaderInterface):
    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)

    def load_data(self):
        (train_images, _), (test_images, _) = mnist.load_data()
        normalized_train = self.normalize_input(train_images)
        flattened_normalized_train = self.flatten_input(normalized_train)
        train_images_noisy = self.add_noise(flattened_normalized_train)

        normalized_test = self.normalize_input(test_images)
        flattened_normalized_test = self.flatten_input(normalized_test)
        test_images_noisy = self.add_noise(flattened_normalized_test)
        
        return {'train_data': train_images_noisy, 
                'train_target': flattened_normalized_train, 
                'test_data':test_images_noisy,
                'test_target':flattened_normalized_test}

    def add_noise(self, input_data, noise_factor=0.5): 
        noise_factor = 0.5
        input_data_noisy = input_data + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=input_data.shape)
        input_data_noisy = np.clip(input_data_noisy, 0., 1.)
        return input_data_noisy

    def flatten_input(self, input_data): 
        flattened_input = input_data.reshape((len(input_data), np.prod(input_data.shape[1:])))
        self.flattened_input_size = flattened_input.shape[1]
        return flattened_input

    def normalize_input(self, input_data): 
        '''Normalize the images by dividing by the maximum pixel value'''
        return input_data.astype('float32') / 255.