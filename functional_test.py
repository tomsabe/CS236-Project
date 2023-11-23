
import numpy as np
import random
from tqdm import tqdm

from load_mnist import load_mnist_binarized
from transformations import GF2LinearLayer, TransformationFlow
from utils import hamming_distance, print_arrays_side_by_side

TARGET_DIGIT = 6
BATCH_SIZE = 1000
INT_TYPE = np.int32
MAX_ONES = 3
TOTAL_TRIES = 30000

if __name__ == '__main__':

    #Step One: Load MNIST into Input Block
    mnist_images, labels = load_mnist_binarized()
    target_images = np.array([image for image, label in zip(mnist_images,labels) if label == TARGET_DIGIT],dtype=INT_TYPE)
    train_indexes = np.random.choice(len(target_images),BATCH_SIZE,replace=False)
    input_block = target_images[train_indexes]
    print(f"Created input block of shape: {input_block.shape}")

    #Step Two: Create the Target Block
    target_block = np.zeros_like(input_block, dtype=INT_TYPE)
    for i in range(BATCH_SIZE):
        target_block[i,13:19,13:19] = 1
    assert np.sum(target_block) == 36*BATCH_SIZE

    #Step Three: Initialize the Flow and the Loss
    current_block = input_block
    current_loss = hamming_distance(current_block,target_block)
    print(f"Current loss is {current_loss} (avg == {int(current_loss/BATCH_SIZE)})")

    #Step Four: Greedy Layer-Wise Optimization
    flow = TransformationFlow()
    for i in tqdm(range(TOTAL_TRIES)):
        #make a random transformation
        num_tri_ones = random.randint(1,MAX_ONES)
        num_bias_ones = random.randint(1,MAX_ONES)
        try_transform = GF2LinearLayer.random_invertible(num_tri_ones=num_tri_ones,num_bias_ones=num_bias_ones,transpose=False)
        #transform all samples in batch
        try_block = try_transform.forward(current_block)
        #recalculate the loss
        try_loss = hamming_distance(try_block,target_block)
        #if loss is lower then add the transformation to the flow
        if try_loss < current_loss:
            flow.append(try_transform)
            current_loss = try_loss
            current_block = try_block
            print(f"Layer: {flow.num_layers()}\tLoss: {current_loss}\tAvgLoss: {current_loss/BATCH_SIZE}")

    #Step Five: Display some outputs
    sample = random.randint(0,BATCH_SIZE-1)
    #1 print first and last layers
    print_arrays_side_by_side(input_block[sample,:,:],current_block[sample,:,:],character_mode=False)
    #2 now start with the final layer and show the inverse
    start = current_block[sample,:,:]
    backward = flow.backward(start)
    print_arrays_side_by_side(start,backward,character_mode=False)
    #3 the pure signal
    signal = target_block[0,:,:]
    backward = flow.backward(signal)
    print_arrays_side_by_side(signal,backward,character_mode=False)
    #4 probabalistic model of all 6's
    sixes_prob = np.mean(current_block, axis=0)
    for i in range(10):
        draw = np.random.binomial(1, sixes_prob)
        print_arrays_side_by_side(draw, flow.backward(draw),character_mode=False)
    flow.save(f'flow-{id(flow)}.pkl')
