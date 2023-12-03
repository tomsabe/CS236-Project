
import numpy as np #TBD: Try Galois library
import numba
import random
from tqdm import tqdm

from load_mnist import load_mnist_binarized
from transformations import GF2LinearLayer, TransformationFlow
from utils import hamming_distance, print_arrays_side_by_side
from utils import estimate_batch_rle_complexity
from utils import square_target, batch_hamming_target
from utils import log_loss
from utils import plot_digit_grid
from utils import INT_TYPE

TARGET_DIGITS = [6,7] #pick two digits
MAX_BATCH_SIZE = 100
TOTAL_TRIES = 20000
TRY_PURE_BIAS = True
TRY_GF2 = True
MAX_TRI_ONES = 3 #somewhere around 3 seems like right balance of variety w/o wasting a lot of iterations
MAX_BIAS_ONES = 0 #bias might be helpful for escaping local minima
TRANSPOSE_LAYERS = False
FIRST_ABLATION = 0
ABLATE_THRESHOLD = 0.25
STOPPING_CRITERIA = 3000
TRY_SCRAMBLE = False

PIXEL_LAMBDA = 1 # use 2^x
HAMMING_LAMBDA = 0 # use 2^x

# All / nearly all hamming works pretty well 
# Similar log loss for all / nearly all pixel
# 1 pixel + 1 hamming : 
# try methods until convergence

@numba.njit
def loss_function(block, hamming_target):
    '''Loss function used in training loop'''
    loss = 0
    if PIXEL_LAMBDA:
        loss += PIXEL_LAMBDA * np.sum(block,dtype=INT_TYPE)
    if HAMMING_LAMBDA:
        loss += HAMMING_LAMBDA * hamming_distance(block,hamming_target)
    return loss

if __name__ == '__main__':
    #Step 1: Load MNIST
    mnist_images, labels = load_mnist_binarized()
    # get all images of TARGET_DIGITS
    target_indexes = []
    for i in range(len(mnist_images)):
        if labels[i] in TARGET_DIGITS:
            target_indexes.append(i)
    target_images=np.array(mnist_images[target_indexes])
    target_labels=labels[target_indexes]
    # get training and validation sets
    total_indices = np.arange(len(target_images))
    np.random.shuffle(total_indices)
    split_point = int(len(total_indices) * 0.8)
    train_batch_size = min(split_point,MAX_BATCH_SIZE)
    train_indices = total_indices[:train_batch_size]
    val_indices = total_indices[split_point:]
    # creating training and validation sets
    train_images = target_images[train_indices]
    train_labels = target_labels[train_indices]
    val_images = target_images[val_indices]
    val_labels = target_labels[val_indices]
    batch_size = len(train_images)

    #Step 2: Create a Batch
    input_block = train_images
    print(f"Created input block of shape: {input_block.shape}")
    hamming_target = batch_hamming_target(input_block,train_labels)
    #inspect some hamming targets
#    for i in range(10):
#        print_arrays_side_by_side(input_block[i],hamming_target[i])

    #Step 3: Initialize the Flow and Loss
    new_train_labels = train_labels.copy() #we need original and a copy 
    current_block = input_block
    current_loss = loss_function(current_block, hamming_target)
    print(f"Current loss is {current_loss} (avg == {int(current_loss/batch_size)})")

    #Step 4: Greedy Layer-Wise Optimization
    flow = TransformationFlow()
    current_max_tri_ones = MAX_TRI_ONES
    current_max_bias_ones = MAX_BIAS_ONES
    last_success=0
    for i in tqdm(range(TOTAL_TRIES)):
        # first look for lossless opportunity to convert 1's to 0's
        if TRY_PURE_BIAS:
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
        # next try a lossess GF2 transformation
        if TRY_GF2:
            #make a random invertible transformation
            num_tri_ones = random.randint(0,current_max_tri_ones)
            num_bias_ones = random.randint(0,current_max_bias_ones)
            try_transform = GF2LinearLayer.random_invertible(num_tri_ones=num_tri_ones,num_bias_ones=num_bias_ones,transpose=TRANSPOSE_LAYERS)
            #try it by transforming the batch
            trial_block = try_transform.forward(current_block)
            #recalculate the loss
            trial_loss = loss_function(trial_block,hamming_target)
            #if trial is a success then add the transformation to the flow
            if trial_loss < current_loss:
                flow.append(try_transform)
                current_loss = trial_loss
                current_block = trial_block
                last_success=i
                print(f"Layer: {flow.num_layers()}\tLoss: {current_loss}\tAvgLoss: {int(current_loss/batch_size)}\t"\
                    f"T1: {num_tri_ones}\tB1:{num_bias_ones}")
        # after a while look for lossy compression - this is not reversible 
        if i>=FIRST_ABLATION and ABLATE_THRESHOLD:
            mask = (np.mean(current_block, axis=0) >= ABLATE_THRESHOLD).astype(INT_TYPE)
            old_sum = np.sum(current_block)
            current_block *= mask #zero out the less informative (x,y)'s
            new_sum = np.sum(current_block)
            if old_sum > new_sum:
                print(f"Ablated {old_sum-new_sum} bits.")
        # check the stopping criteria
        if i-last_success > STOPPING_CRITERIA:
            if TRY_SCRAMBLE:
                #make a random invertible transformation
                num_tri_ones = random.randint(0,current_max_tri_ones)
                num_bias_ones = random.randint(0,current_max_bias_ones)
                try_transform = GF2LinearLayer.random_invertible(num_tri_ones=num_tri_ones,num_bias_ones=num_bias_ones,transpose=TRANSPOSE_LAYERS)
                #try it by transforming the batch
                trial_block = try_transform.forward(current_block)
                #recalculate the loss
                trial_loss = loss_function(trial_block,hamming_target)
                #add the transformation to the flow
                if True:
                    flow.append(try_transform)
                    current_loss = trial_loss
                    current_block = trial_block
                    last_success=i
                    print(f"Scramble Layer: {flow.num_layers()}\tLoss: {current_loss}\tAvgLoss: {int(current_loss/batch_size)}\t"\
                        f"T1: {num_tri_ones}\tB1:{num_bias_ones}")
            else:
                break

    #Step 5: Display some outputs
    sample = random.randint(0,batch_size-1)
    #1 print first and last layers
    print_arrays_side_by_side(input_block[sample,:,:],current_block[sample,:,:],character_mode=False)
    #2 now start with the final layer and show the inverse
    start = current_block[sample,:,:]
    backward = flow.backward(start)
    print_arrays_side_by_side(start,backward,character_mode=False)
    #3 generative model
    input_prob = np.mean(input_block, axis=0)
    sixes_block = current_block[new_train_labels == 6]
    sevens_block = current_block[new_train_labels == 7]
    sixes_prob = np.mean(sixes_block, axis=0)
    sevens_prob = np.mean(sevens_block, axis=0)
    # plot heatmaps
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import Reds, Blues
    from matplotlib.gridspec import GridSpec
    # normalized the color maps
    norm = Normalize(vmin=0, vmax=1)
    # set up gridspec layout
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
    plt.show()

    # split validation set based on the digit
    val_images_6 = val_images[val_labels == 6]
    val_images_7 = val_images[val_labels == 7]

    # generate samples for "6" and "7"
    draw_samples_6 = np.random.binomial(1, sixes_prob, size=val_images_6.shape)
    draw_samples_7 = np.random.binomial(1, sevens_prob, size=val_images_7.shape)

    # calculate log loss for "6"
    losses_6 = []
    for i in range(len(val_images_6)):
        loss = log_loss(val_images_6[i], draw_samples_6[i])
        losses_6.append(loss)
    avg_log_loss_6 = np.mean(losses_6)

    # calculate log loss for "7"
    losses_7 = []
    for i in range(len(val_images_7)):
        loss = log_loss(val_images_7[i], draw_samples_7[i])
        losses_7.append(loss)
    avg_log_loss_7 = np.mean(losses_7)

    # use average pixel probability baseline
    avg_pixel_prob_6 = np.mean(train_images[train_labels == 6])
    avg_pixel_prob_7 = np.mean(train_images[train_labels == 7])
    avg_pixel_prob_baseline_6 = np.random.binomial(1, avg_pixel_prob_6, size=val_images_6.shape)
    avg_pixel_prob_baseline_7 = np.random.binomial(1, avg_pixel_prob_7, size=val_images_7.shape)

    # Calculate log loss for baseline
    avg_pixel_prob_loss_6 = np.mean([log_loss(val_images_6[i], avg_pixel_prob_baseline_6[i]) for i in range(len(val_images_6))])
    avg_pixel_prob_loss_7 = np.mean([log_loss(val_images_7[i], avg_pixel_prob_baseline_7[i]) for i in range(len(val_images_7))])

    print(f"Naive Avg Log Loss for '6': {avg_pixel_prob_loss_6}\tNaive Avg Log Loss for '7': {avg_pixel_prob_loss_7}")
    print(f"Model Avg Log Loss for '6': {avg_log_loss_6}\tModel Avg Log Loss for '7': {avg_log_loss_7}")

    # Show some examples of generated '6' and '7'
    plot_digit_grid(flow.backward(draw_samples_6), "Generated 6's")
    plot_digit_grid(flow.backward(draw_samples_7), "Generated 7's")

    #4 save the flow
    flow.latent_distribution = sixes_prob #save alongside the flow
    filename = f'flow-{id(flow)}.pkl' 
    flow.save(filename)
    print(f"Saved {filename}")


#            # Calculate the sum for each item in the batch
#            sums = np.sum(current_block, axis=(1, 2))
#            # Find the index of the item with the lowest sum
#            lowest_sum_index = np.argmin(sums)
#            # Remove the item with the lowest sum
#            current_block = np.delete(current_block, lowest_sum_index, axis=0)
#            hamming_target = np.delete(hamming_target, lowest_sum_index, axis=0)
#            new_train_labels = np.delete(new_train_labels, lowest_sum_index, axis=0)
#            batch_size -= 1
#            last_success = i

#            if ABLATE_THRESHOLD:
#                #mask the least informative (x,y) that is non-zero
#                mean_values = np.mean(current_block, axis=0)
#                mean_values[mean_values == 0] = np.nan
#                min_mean_pos = np.unravel_index(np.nanargmin(mean_values), mean_values.shape)
#                zero_mask = np.ones_like(current_block[0], dtype=INT_TYPE)
#                zero_mask[min_mean_pos] = 0
#                current_block *= zero_mask
#                print(f"Masked the least informative non-zero (x,y) in the batch.")
#                last_success=i #reset the counter to give it another try
#            else:
#                break
