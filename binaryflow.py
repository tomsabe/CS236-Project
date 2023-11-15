import numpy as np
import random

from load_mnist import load_mnist_binarized
from transformations import Transformation

class BitFlow:
    def __init__(self, first_layer: np.ndarray, label=None, mnist_label=None):
        self.block = np.zeros((1, 32, 32), dtype=int)
        self.block[0, :, :] = first_layer
        self.transformations = [] #sequence of Transformations
        self.label = label #matrix version of label
        self.mnist_label = mnist_label #int version of label

    def forward(self, layer: np.array) -> np.array:
        '''Apply all transformations to the specified layer'''
        for i in range(0,len(self.transformations)):
            layer = self.transformations[i].forward(layer)
        return layer

    def backward(self, layer: np.array) -> np.array:
        '''Apply all inverse transformations to the specified layer'''
        for i in range(1,len(self.transformations)+1):
            layer = self.transformations[-i].backward(layer)
        return layer

    def add_transformation(self, transformation: Transformation):
        '''Add a transformation and a new layer to the Flow.'''
        self.transformations.append(transformation)
        new_layer = transformation.forward(self.block[-1, :, :])
        new_layer = np.reshape(new_layer, (1,32,32))
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
                label_matrix = np.ones((1,32,32), dtype=int)
            else:
                label_matrix = np.zeros((1,32, 32), dtype=int)
            bitflow = BitFlow(img)
            bitflow.label = label_matrix
            bitflow.mnist_label = int(label)
            self.batch.append(bitflow)
        return self.batch
   
    def get_sample_bitflow(self):
        return random.choice(self.batch)
