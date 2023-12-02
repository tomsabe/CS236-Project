import numpy as np
import random

from load_mnist import load_mnist_binarized
from transformations import Transformation

class BitFlow:
    def __init__(self, first_layer: np.ndarray, label=None, mnist_label=None):
        self.block = np.zeros((1, 32, 32), dtype=np.int32)
        self.block[0, :, :] = first_layer
        self.label = label #matrix version of label
        self.mnist_label = mnist_label #int version of label

    def apply_transformation(self, transformation: Transformation):
        '''Apply a transformation, adding a new layer.'''
        new_layer = transformation.forward(self.block[-1, :, :])
        new_layer = np.reshape(new_layer, (1,32,32)) #TBD: Eliminate this step
        self.block = np.concatenate((self.block, new_layer), axis=0)
        return self.block

    def try_transformation(self, transformation: Transformation):
         '''Try out the outcome of applying a transformation, but do not actually apply it.'''
         return transformation.forward(self.block[-1, :, :])

    def show_layer(self, n: int):
        '''Prints the nth layer as a 32x32 grid of 1's and 0's'''
        if n < 0 or n >= self.block.shape[0]:
            raise ValueError(f"Layer {n} is out of bounds.")        
        for row in self.block[n, :, :]:
            print(''.join(map(str, row)))

class MNISTBitFlowBatcher:
    def __init__(self, batch_size=512):
        self.batch_size = batch_size
        self.train_images, self.train_labels = load_mnist_binarized()
        self.batch = []

    def get_batch(self, target_label=0):
        '''Get a batch of binarized MNIST digits.'''      
        # Randomly select indices for the batch
        indices = np.random.choice(len(self.train_images), self.batch_size, replace=False)
        batch_images = self.train_images[indices]
        batch_labels = self.train_labels[indices]      
        # Convert images and labels to BitFlows
        for img, label in zip(batch_images, batch_labels):
            if label == target_label:
                label_matrix = np.ones((1,32,32), dtype=np.int32)
            else:
                label_matrix = np.zeros((1,32, 32), dtype=np.int32)
            bitflow = BitFlow(img)
            bitflow.label = label_matrix
            bitflow.mnist_label = int(label)
            self.batch.append(bitflow)
        return self.batch
   
    def get_sample_bitflow(self):
        return random.choice(self.batch)
