import numpy as np
from pprint import pprint
from tqdm import tqdm

from binaryflow import MNISTBitFlowBatcher
from transformations import GF2LinearLayer
from utils import estimate_rle_length, hamming_distance_a_b, print_arrays_side_by_side

#SIGNAL = np.triu(np.ones((32, 32), dtype=int))
SIGNAL = np.zeros((32,32),dtype=int)
for i in range(8,24):
    for j in range(8,24):
        SIGNAL[i,j] = 1

class GreedyOptimizer:
    def __init__(self, batch_size=512):
        self.bitflow_batcher = MNISTBitFlowBatcher(batch_size=batch_size)

    def hamming_loss(self, samples):
        '''Calculate the Hamming loss function'''
#        for sample in samples:
#            print_arrays_side_by_side(sample,SIGNAL)
#            print(f"Hamming: {hamming_distance_a_b(sample,SIGNAL)}")
#            input("ok")
        loss = sum([hamming_distance_a_b(sample,SIGNAL) for sample in samples])/len(samples)
        return loss

    def rlel_loss(self, samples):
        '''Calculate RLEL loss function'''
        loss = sum([estimate_rle_length(sample) for sample in samples])/len(samples)
        return loss

    def optimize(self, iterations=1000, display_every=10):
        # Get a batch
        batch = self.bitflow_batcher.get_batch()
        #optional : set initial loss based on starting layer
#        label_6_samples = [bc.block[-1] for bc in batch if bc.mnist_label == 6]
#        other_samples = [bc.block[-1] for bc in batch if bc.mnist_label != 6]
#        loss = self.rlel_loss(label_6_samples)
        # better: 
        loss = 10^6 #set crazy high initial loss
        layers = 1
#        total_distance = sum([bernoulli_distance(bc.block[:,:,-1]) for bc in batch]) \
#            -sum([estimate_rle_length(bc.block[:,:,-1]) for bc in batch])
        for iteration in tqdm(range(iterations)):
            # Get a new random invertible transformation
            new_transform = GF2LinearLayer.random_invertible()
            try_samples = [bc.try_transformation(new_transform) for bc in batch if bc.mnist_label == 6]
            new_loss = self.hamming_loss(try_samples)
#            new_loss = sum([bernoulli_distance(bc.try_transformation(new_transform)) for bc in batch]) \
#                -sum([estimate_rle_length(bc.try_transformation(new_transform)) for bc in batch])
            # If the new distance is smaller, add the layer
            if new_loss < loss or iteration < 3:
                for bc in batch:
                    bc.add_transformation(new_transform) #actually we only need one transformation flow
                layers += 1
                loss = new_loss

            # Display progress
            if iteration % display_every == 0:
                print(f"Iteration: {iteration} | Loss: {loss} | Layers: {layers} ")
        print(f"Iteration: {iteration} | Loss: {loss} | Layers: {layers} ")
        return batch

