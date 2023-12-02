
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
from utils import INT_TYPE

TARGET_DIGITS = [6,7] #pick two digits
MAX_BATCH_SIZE = 100
MAX_TRI_ONES = 3 #somewhere around 3 seems like right balance of variety w/o wasting a lot of iterations
MAX_BIAS_ONES = 1 #bias seems to be helpful for escaping local minima
TOTAL_TRIES = 20000
PIXEL_LAMBDA = 1 # use 2^x
HAMMING_LAMBDA = 16 # use 2^x

@numba.njit
def loss_function(block, hamming_target):
    '''Loss function used in training loop'''
    pixels = np.sum(block,dtype=INT_TYPE)
#    complexity = estimate_batch_rle_complexity(block)+estimate_batch_rle_complexity(np.transpose(block,(0,2,1)))
    hamming_loss = hamming_distance(block,hamming_target)
    return hamming_loss + pixels
#    return PIXEL_LAMBDA*pixels+HAMMING_LAMBDA*hamming_loss
#    return -complexity
#    return hamming_loss
#    return pixels * PIXEL_LAMBDA + hamming_loss * HAMMING_LAMBDA - complexity

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
#    target_images = np.array([image for image, label in zip(mnist_images,labels) if label in TARGET_DIGITS],dtype=INT_TYPE)
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

    #Step 3: Initialize the Flow and Loss
    current_block = input_block
    current_loss = loss_function(current_block, hamming_target)
    print(f"Current loss is {current_loss} (avg == {int(current_loss/batch_size)})")

    #Step 4: Greedy Layer-Wise Optimization
    flow = TransformationFlow()
    current_max_tri_ones = MAX_TRI_ONES
    current_max_bias_ones = MAX_BIAS_ONES
    for i in tqdm(range(TOTAL_TRIES)):
#       Likely more efficient to dynamically adjust maximums as training converges:
#        if i==50:
#            current_max_tri_ones = min(1,MAX_TRI_ONES)
#            current_max_bias_ones = min(1,MAX_BIAS_ONES)
        #make a random invertible transformation
        num_tri_ones = random.randint(0,current_max_tri_ones)
        num_bias_ones = random.randint(0,current_max_bias_ones)
        try_transform = GF2LinearLayer.random_invertible(num_tri_ones=num_tri_ones,num_bias_ones=num_bias_ones)#,transpose=True)
        #try it by transforming the batch
        trial_block = try_transform.forward(current_block)
        #recalculate the loss
        trial_loss = loss_function(trial_block,hamming_target)
        #if trial is a success then add the transformation to the flow
        if trial_loss < current_loss:
            flow.append(try_transform)
            current_loss = trial_loss
            current_block = trial_block
            print(f"Layer: {flow.num_layers()}\tLoss: {current_loss}\tAvgLoss: {int(current_loss/batch_size)}\t"\
                  f"T1: {num_tri_ones}\tB1:{num_bias_ones}")
            #add a pure bias layer if helpful
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
    sixes_block = current_block[train_labels == 6]
    sevens_block = current_block[train_labels == 7]

    sixes_prob = np.mean(sixes_block, axis=0)
    sevens_prob = np.mean(sevens_block, axis=0)

    # Plotting the heatmaps
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import Reds, Blues
    from matplotlib.gridspec import GridSpec

    # Normalization for the color maps
    norm = Normalize(vmin=0, vmax=1)

    # Set up a gridspec layout for the heatmap and colorbars
    plt.figure(figsize=(10, 6))
    gs = GridSpec(1, 3, width_ratios=[30, 1, 1])

    # Create the Axes for the heatmap
    ax_heatmap = plt.subplot(gs[0])

    # Overlay the heatmaps for "6" and "7"
    im_6 = ax_heatmap.imshow(sixes_prob, cmap=Blues, alpha=0.5, norm=norm)
    im_7 = ax_heatmap.imshow(sevens_prob, cmap=Reds, alpha=0.5, norm=norm)

    # Add a title and a legend
    ax_heatmap.set_title('Overlayed Probability Maps for "6" and "7"')
    ax_heatmap.legend(['"6" Probability', '"7" Probability'], loc='upper right')

    # Create the Axes for the colorbars
    ax_cbar_6 = plt.subplot(gs[1])
    ax_cbar_7 = plt.subplot(gs[2])

    # Create colorbars
    plt.colorbar(im_6, cax=ax_cbar_6).set_label('Probability for "6"')
    plt.colorbar(im_7, cax=ax_cbar_7).set_label('Probability for "7"')

    plt.show()


    # Splitting the validation set based on the digit
    val_images_6 = val_images[val_labels == 6]
    val_images_7 = val_images[val_labels == 7]

    # Generate separate samples for "6" and "7"
    draw_samples_6 = np.random.binomial(1, sixes_prob, size=val_images_6.shape)
    draw_samples_7 = np.random.binomial(1, sevens_prob, size=val_images_7.shape)

    # Calculate log loss for "6"
    losses_6 = []
    for i in range(len(val_images_6)):
        loss = log_loss(val_images_6[i], draw_samples_6[i])
        losses_6.append(loss)
    avg_log_loss_6 = np.mean(losses_6)

    # Calculate log loss for "7"
    losses_7 = []
    for i in range(len(val_images_7)):
        loss = log_loss(val_images_7[i], draw_samples_7[i])
        losses_7.append(loss)
    avg_log_loss_7 = np.mean(losses_7)

    # Average pixel probability baseline
    avg_pixel_prob_6 = np.mean(train_images[train_labels == 6])
    avg_pixel_prob_7 = np.mean(train_images[train_labels == 7])

    avg_pixel_prob_baseline_6 = np.random.binomial(1, avg_pixel_prob_6, size=val_images_6.shape)
    avg_pixel_prob_baseline_7 = np.random.binomial(1, avg_pixel_prob_7, size=val_images_7.shape)

    # Calculate log loss for the average pixel probability baseline
    avg_pixel_prob_loss_6 = np.mean([log_loss(val_images_6[i], avg_pixel_prob_baseline_6[i]) for i in range(len(val_images_6))])
    avg_pixel_prob_loss_7 = np.mean([log_loss(val_images_7[i], avg_pixel_prob_baseline_7[i]) for i in range(len(val_images_7))])

    # Print log loss for each digit
    print(f"Naive Avg Log Loss for '6': {avg_pixel_prob_loss_6}\tNaive Avg Log Loss for '7': {avg_pixel_prob_loss_7}")
    print(f"Model Avg Log Loss for '6': {avg_log_loss_6}\tModel Avg Log Loss for '7': {avg_log_loss_7}")

    # Show some examples of generated '6' and '7'
    for i in range(5):  # Show 5 examples for each
        draw_6 = draw_samples_6[i]
        draw_7 = draw_samples_7[i]
        print("Digit 6:")
        print_arrays_side_by_side(draw_6, flow.backward(draw_6), character_mode=False)
        print("Digit 7:")
        print_arrays_side_by_side(draw_7, flow.backward(draw_7), character_mode=False)

    #4 save the flow
    flow.latent_distribution = sixes_prob #save alongside the flow
    filename = f'flow-{id(flow)}.pkl' 
    flow.save(filename)
    print(f"Saved {filename}")

'''
    naive_baseline = np.random.binomial(1, input_prob, size=val_images.shape)
    draw_samples = np.random.binomial(1, sixes_prob, size=val_images.shape)
    # calculate average log loss of the naive baseline
    losses = []
    for i in range(len(val_images)):
        loss = log_loss(val_images[i], naive_baseline[i])
        losses.append(loss)
    baseline_avg_log_loss = np.mean(losses)
    # calculate average log loss of the model
    losses = []
    for i in range(len(val_images)):
        loss = log_loss(val_images[i], draw_samples[i])
        losses.append(loss)
    model_avg_log_loss = np.mean(losses)
    # print parameter count and log loss
    flow.parameter_count()
    print(f"Baseline log loss: {baseline_avg_log_loss}\tModel log loss: {model_avg_log_loss}")
    # show some examples
    for i in range(10):
        draw = draw_samples[i]
        print_arrays_side_by_side(draw, flow.backward(draw),character_mode=False)
'''