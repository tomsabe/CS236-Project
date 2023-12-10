'''CS236 Final Project - Binary Generative Flows - Demonstration'''
'''Tom Saberhagen tsabe@stanford.edu'''
'''December 11, 2023'''

# ISSUES and TBD:
# Binary operations are implemented using Python, Numpy for demonstration purposes
#   (could be further optimized with lower level language)

import numpy as np
import numba
import random
from tqdm import tqdm
import time

from load_mnist import load_mnist_binarized
from transformations import GF2LinearLayer, TransformationFlow
from utils import hamming_distance, print_arrays_side_by_side
from utils import batch_hamming_target
from utils import log_loss
from utils import plot_digit_grid
from utils import INT_TYPE
from utils import estimate_batch_rle_complexity

TARGET_DIGITS = [6,7] #use of 6 and 7 is hardcoded

#Hyperparameters:
MAX_BATCH_SIZE = 100
MAX_ITERATIONS = 20000
STOPPING_CRITERIA = 3000 #stop if no optimizing action found after this # of tries

# training loop:
TRY_PURE_BIAS = True
TRY_GF2 = True
MAX_TRI_ONES = 3 #somewhere around 3 seems like right balance of variety w/o wasting a lot of iterations
MAX_BIAS_ONES = 0 #including bias term in WX+B GF(2) linear transformation hasn't been helpful
TRANSPOSE_LAYERS = True #Transposing X has little log loss effect but may help digits look a little better
FIRST_ABLATION = 0 #Delaying the ablation of less informative (x,y) does not seem to be helpful for our MNIST data
ABLATE_THRESHOLD = 0.35 #0.35 works with MNIST data -- amenable to relatively high ablation

# loss function:
BITCOUNT_LAMBDA = 2**0 # weighting factor for loss function; use 2^x so amenable to bitshift
RLE_LENGTH_LAMBDA = 0 #weighting factor for RLE-length function; use 2^x so amenable to bitshift
HAMMING_LAMBDA = 0 # weighting factor for loss function; use 2^x so amenable to bitshift

#@numba.njit
def loss_function(block, hamming_target):
    '''Loss function used in training loop'''
    loss = 0
    if BITCOUNT_LAMBDA: #2^2 or 2^3 may be a good choice when incorporating the difference measures
        loss += BITCOUNT_LAMBDA * np.sum(block,dtype=INT_TYPE)
    if RLE_LENGTH_LAMBDA:
        #estimate for each layer and its transpose
        rle_measure = 0
        for i in range(len(block)):
            rle_measure += np.abs(estimate_batch_rle_complexity(block[i])-512).astype(INT_TYPE)
            rle_measure += np.abs(estimate_batch_rle_complexity(np.transpose(block[i],(1,0)).astype(INT_TYPE))-512).astype(INT_TYPE)
        loss += RLE_LENGTH_LAMBDA * rle_measure
    if HAMMING_LAMBDA:
        loss += HAMMING_LAMBDA * hamming_distance(block,hamming_target)
    return loss

if __name__ == '__main__':

    # Step 1: Load MNIST
    mnist_images, labels = load_mnist_binarized()
    print(f"Loaded {len(mnist_images)} binarized MNIST images.")

    # get all images of TARGET_DIGITS
    target_indexes = []
    for i in range(len(mnist_images)):
        if labels[i] in TARGET_DIGITS:
            target_indexes.append(i)
    target_images=np.array(mnist_images[target_indexes])
    target_labels=labels[target_indexes]
    print(f"Found {len(target_images)} in target set: {TARGET_DIGITS}")

    # get training and validation sets
    total_indices = np.arange(len(target_images))
    np.random.shuffle(total_indices)
    split_point = int(len(total_indices) * 0.8)
    train_batch_size = min(split_point,MAX_BATCH_SIZE)
    train_indices = total_indices[:train_batch_size]
    val_indices = total_indices[split_point:]
    train_images = target_images[train_indices]
    train_labels = target_labels[train_indices]
    val_images = target_images[val_indices]
    val_labels = target_labels[val_indices]
    batch_size = len(train_images)
    print(f"Selected {len(train_images)} training images and {len(val_images)} validation images.")

    # Step 2: Create a Batch
    input_block = train_images
    print(f"Created input batch of shape: {input_block.shape}")
    # create hamming targets for the batch
    hamming_target = batch_hamming_target(input_block,train_labels)
    # optional - if experimenting with hamming targets, may want to inspect some samples here:
#    for i in range(10):
#        print_arrays_side_by_side(input_block[i],hamming_target[i])

    # Step 3: Initialize the block, the loss, and the flow
#    new_train_labels = train_labels.copy() #we need original and a copy 
    current_block = input_block
    current_loss = loss_function(current_block, hamming_target)
    print(f"Starting loss is {current_loss} (avg == {int(current_loss/batch_size)})")
    flow = TransformationFlow()
    current_max_tri_ones = MAX_TRI_ONES #'current' variable enables dynamic adjustment experiments
    current_max_bias_ones = MAX_BIAS_ONES

    #Step 4: Greedy Layer-Wise Optimization
    last_success=0
    for i in tqdm(range(MAX_ITERATIONS)):

        if TRY_PURE_BIAS: #lossless opportunities to flip 1's to 0's and reduce bitcount
            pure_bias = (np.mean(current_block, axis=0) > 0.5).astype(INT_TYPE)
            pure_bias_xform = GF2LinearLayer.bias_only(pure_bias)
            trial_block = pure_bias_xform.forward(current_block)
            trial_loss = loss_function(trial_block,hamming_target)
            if trial_loss < current_loss:
                flow.append(pure_bias_xform)
                current_loss = trial_loss
                current_block = trial_block
                print(f"Layer: {flow.num_layers()}\tLoss: {current_loss}\tAvgLoss: {int(current_loss/batch_size)}\t"\
                    f"BiasSum:{np.sum(pure_bias)}")
                
        if TRY_GF2: #try lossless "WX + B" transformation in GF(2)
            #make a random invertible transformation
            num_tri_ones = random.randint(0,current_max_tri_ones)
            num_bias_ones = random.randint(0,current_max_bias_ones) #we found good results without Bias (current_max_bias_ones = 0)
            try_transform = GF2LinearLayer.random_invertible(num_tri_ones=num_tri_ones,num_bias_ones=num_bias_ones,transpose=TRANSPOSE_LAYERS)
            #try it out by transforming the batch
            trial_block = try_transform.forward(current_block)
            #recalculate the trial loss
            trial_loss = loss_function(trial_block,hamming_target)
            #if trial loss is lower than current loss then add the transformation to the flow
            if trial_loss < current_loss:
                flow.append(try_transform)
                current_loss = trial_loss
                current_block = trial_block
                last_success=i
                print(f"Layer: {flow.num_layers()}\tLoss: {current_loss}\tAvgLoss: {int(current_loss/batch_size)}\t"\
                    f"T1: {num_tri_ones}\tB1:{num_bias_ones}")
                
        if i>=FIRST_ABLATION and ABLATE_THRESHOLD: #lossy compression - but speeds training and gets smaller model
            mask = (np.mean(current_block, axis=0) >= ABLATE_THRESHOLD).astype(INT_TYPE) #1's if > threshold
            old_sum = np.sum(current_block) #only for recordkeeping/display purposes
            current_block *= mask #zero out the less informative (x,y)'s
            new_sum = np.sum(current_block) #only for recordkeeping/display purposes
            if old_sum > new_sum:
                print(f"Ablated {old_sum-new_sum} bits.")

        if i-last_success > STOPPING_CRITERIA: #check the stopping criteria
            break

    #Step 5: Display some outputs
    for i in range(3):
        print(f"Displaying sample {i+1} of 3.")
        sample = random.randint(0,batch_size-1)
        #1 print first and last layers
        print_arrays_side_by_side(input_block[sample,:,:],current_block[sample,:,:],character_mode=False)
        #2 now, start with the final layer and show its inverse
        start = current_block[sample,:,:]
        backward = flow.backward(start)
        print_arrays_side_by_side(start,backward,character_mode=False)
    #3 generative model
    input_prob = np.mean(input_block, axis=0)
    sixes_block = current_block[train_labels == 6] #new_train_labels
    sevens_block = current_block[train_labels == 7] #new_train_labels
    sixes_prob = np.mean(sixes_block, axis=0)
    sevens_prob = np.mean(sevens_block, axis=0)
    print("Displaying MNIST sixes and sevens latent space probabilities.")
    print_arrays_side_by_side(sixes_prob,sevens_prob,character_mode=False)
    latent_sample_6 = np.random.binomial(1, sixes_prob, size=sixes_prob.shape)
    latent_sample_7 = np.random.binomial(1, sevens_prob, size=sevens_prob.shape)
    print("Displaying a sample 6 and a sample 7 in latent space.")
    print_arrays_side_by_side(latent_sample_6,latent_sample_7,character_mode=False)
    # plot overlaid probabilities
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import Reds, Blues
    from matplotlib.gridspec import GridSpec
    norm = Normalize(vmin=0, vmax=1)
    plt.figure(figsize=(10, 6))
    gs = GridSpec(1, 3, width_ratios=[30, 1, 1])
    ax_heatmap = plt.subplot(gs[0])
    im_6 = ax_heatmap.imshow(sixes_prob, cmap=Blues, alpha=0.5, norm=norm)
    im_7 = ax_heatmap.imshow(sevens_prob, cmap=Reds, alpha=0.5, norm=norm)
    ax_heatmap.set_title('Overlayed Probability Maps for "6" and "7"')
    ax_heatmap.legend(['"6" Probability', '"7" Probability'], loc='upper right')
    ax_cbar_6 = plt.subplot(gs[1])
    ax_cbar_7 = plt.subplot(gs[2])
    plt.colorbar(im_6, cax=ax_cbar_6).set_label('Probability for "6"')
    plt.colorbar(im_7, cax=ax_cbar_7).set_label('Probability for "7"')
    print("Displaying probability maps for '6' and '7'")
    plt.show()
    # split validation set based on the digit
    val_images_6 = val_images[val_labels == 6]
    val_images_7 = val_images[val_labels == 7]
    # generate samples for "6" and "7"
    start_time = time.time()
    draw_samples_6 = flow.backward(np.random.binomial(1, sixes_prob, size=val_images_6.shape))
    end_time = time.time()
    print(f"Calculated {len(draw_samples_6)} backward flows in {end_time-start_time} seconds.")
    print(f"Average seconds per flow = {(end_time-start_time)/len(draw_samples_6)}")
    start_time = time.time()
    draw_samples_7 = flow.backward(np.random.binomial(1, sevens_prob, size=val_images_7.shape))
    end_time = time.time()
    print(f"Calculated {len(draw_samples_7)} backward flows in {end_time-start_time} seconds.")
    print(f"Average seconds per flow = {(end_time-start_time)/len(draw_samples_7)}")
    # calculate log loss for generated "6" and "7"
    avg_log_loss_6 = np.mean([log_loss(val_images_6[i], draw_samples_6[i]) for i in range(len(val_images_6))])
    avg_log_loss_7 = np.mean([log_loss(val_images_7[i], draw_samples_7[i]) for i in range(len(val_images_7))])
    # calculate average pixel probability baseline
    avg_pixel_prob_6 = np.mean(train_images[train_labels == 6])
    avg_pixel_prob_7 = np.mean(train_images[train_labels == 7])
    avg_pixel_prob_baseline_6 = np.random.binomial(1, avg_pixel_prob_6, size=val_images_6.shape)
    avg_pixel_prob_baseline_7 = np.random.binomial(1, avg_pixel_prob_7, size=val_images_7.shape)
    # calculate log loss for baseline
    avg_pixel_prob_loss_6 = np.mean([log_loss(val_images_6[i], avg_pixel_prob_baseline_6[i]) for i in range(len(val_images_6))])
    avg_pixel_prob_loss_7 = np.mean([log_loss(val_images_7[i], avg_pixel_prob_baseline_7[i]) for i in range(len(val_images_7))])
    print(f"Naive Avg Log Loss for '6': {avg_pixel_prob_loss_6}\tNaive Avg Log Loss for '7': {avg_pixel_prob_loss_7}")
    print(f"Model Avg Log Loss for '6': {avg_log_loss_6}\tModel Avg Log Loss for '7': {avg_log_loss_7}")

    # Show some examples of generated '6' and '7'
    print("Displaying 100 samples of generated 6's")
    plot_digit_grid(draw_samples_6, "Generated 6's")
    print("Displaying 100 samples of generated 7's")
    plot_digit_grid(draw_samples_7, "Generated 7's")

    #4 save the flow
    flow.latent_distribution = {"6":sixes_prob,"7":sevens_prob} #save alongside the flow
    filename = f'models/flow-{id(flow)}.pkl' 
    flow.save(filename)
    print(f"Saved flow as {filename}")
