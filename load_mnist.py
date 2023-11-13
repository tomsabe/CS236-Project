import numpy as np
from sklearn.datasets import fetch_openml
import os

def load_mnist_binarized(cache_path="binarized_mnist.npz"):
    '''Load and binarize the MNIST dataset.'''
    # Check if cached data exists
    if os.path.exists(cache_path):
        # Load from cached file
        with np.load(cache_path) as data:
            images = data['images']
            labels = data['labels']
    else:
        # Fetch and process the data
        mnist = fetch_openml('mnist_784', version=1, parser='auto', cache=True, return_X_y=True)
        images = np.array(mnist[0], dtype=np.uint8).reshape(-1, 28, 28)
        labels = np.array(mnist[1], dtype=np.uint8)
        # Binarize
        images = np.where(images > 127, 1, 0)
        # Add padding to get 32x32 size
        images = np.pad(images, ((0,0), (2,2), (2,2)), 'constant')
        # Save processed data for future use
        np.savez(cache_path, images=images, labels=labels)
    return images, labels
