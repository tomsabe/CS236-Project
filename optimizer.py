# TBD: Does not need to return a batch
#      (batches used for training; result is just the flow)
# TBD: Layer-wise optimizer
# TBD: what is the optimum signal
# TBD: is it better to search shallow or deep? 


import numba
import numpy as np
from pprint import pprint
import random
from tqdm import tqdm

from transformations import TransformationFlow, GF2LinearLayer, Transformation
from utils import estimate_rle_length, hamming_distance_a_b, print_arrays_side_by_side

SIGNAL = np.zeros((32,32),dtype=np.int32)
for i in range(13,19):
    for j in range(13,19):
        SIGNAL[i,j] = 1

NUM_TRIES = 1000
MAX_ONES = 6

@numba.njit
def hamming_loss(samples, signal):
    '''Calculate the Hamming loss of the samples'''
    loss = sum([hamming_distance(sample,signal) for sample in samples])
    return loss

class GreedyOptimizer:
    def __init__(self, batcher):
        self.bitflow_batcher = batcher
        self.signal = SIGNAL
        self.flow = TransformationFlow()

    def rlel_loss(self, samples):
        '''Calculate the RLEL loss of the samples'''
        loss = sum([estimate_rle_length(sample) for sample in samples])
        return loss

    def find_layer_transform(self, samples, current_loss, n_tries = NUM_TRIES):
        final_transform = Transformation()
        final_num_tri_ones = 0
        final_num_bias_ones = 0
        for i in range(n_tries):
            num_tri_ones = random.randint(1,MAX_ONES)
            num_bias_ones = random.randint(1,MAX_ONES)
            try_transform = GF2LinearLayer.random_invertible(num_tri_ones=num_tri_ones,num_bias_ones=num_bias_ones)
            try_samples = [bc.try_transformation(try_transform) for bc in samples]
            new_loss = hamming_loss(try_samples, self.signal)
            #Keep the lowest loss transform
            if new_loss < current_loss:
                final_transform = try_transform
                current_loss = new_loss
                final_num_tri_ones = num_tri_ones
                final_num_bias_ones = num_bias_ones
        if final_num_tri_ones+final_num_bias_ones != 0:
            print(f"n_tri: {final_num_tri_ones}\tn_bias:{final_num_bias_ones} loss: {current_loss}")
        return final_transform, current_loss

    def optimize(self, iterations=1000, display_every=10, mnist_target=6):
        #Reset the flow
        self.flow = TransformationFlow()
        #Use the batcher to get a batch
        batch = self.bitflow_batcher.get_batch()
        target_samples = [sample for sample in batch if sample.mnist_label == mnist_target]
        non_target_samples = [sample for sample in batch if sample.mnist_label != mnist_target]
        current_loss = 1000000 #set crazy high initial loss
        for iteration in tqdm(range(iterations)):
            layer_transform, new_loss = self.find_layer_transform(target_samples, current_loss)
            if new_loss < current_loss: #or iteration < 3: #TBD - alternate scramble mechanism vs. first 3 layers
                self.flow.append(layer_transform)
                for bc in batch:
                    bc.apply_transformation(layer_transform)
                current_loss = new_loss
            # Display progress
            if iteration % display_every == 0:
                print(f"Iteration: {iteration} | Loss: {current_loss} | Layers: {self.flow.num_layers()} ")
        print(f"Iteration: {iteration} | Loss: {current_loss} | Layers: {self.flow.num_layers()} ")
        return batch

