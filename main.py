import numpy as np
import random

from optimizer import GreedyOptimizer
from utils import print_arrays_side_by_side

# Example of usage:
if __name__ == "__main__":
    optimizer = GreedyOptimizer(batch_size=10000)
    batch = optimizer.optimize(iterations=50000, display_every=10000)
    #take a sample from the batch
    sample_bc = random.sample(batch,1)[0]
    while sample_bc.mnist_label != 6:
        sample_bc = random.sample(batch,1)[0]
    #1 print first and last layers
    print_arrays_side_by_side(sample_bc.block[0,:,:],sample_bc.block[-1,:,:],character_mode=False)
    #2 now start with the final layer and show the inverse
    start = sample_bc.block[-1,:,:]
    backward = sample_bc.backward(start)
    print_arrays_side_by_side(start,backward,character_mode=False)
    #3 a 'noised' input
    noisy_start = start ^ np.random.choice([0, 1], size=start.shape, p=[0.99, 0.01])
    backward = sample_bc.backward(noisy_start)
    print_arrays_side_by_side(start,backward,character_mode=False)
    #4 finally, the pure signal
    from optimizer import SIGNAL
    backward = sample_bc.backward(SIGNAL)
    print_arrays_side_by_side(SIGNAL,backward,character_mode=False)


    #UNIT TEST FOR THE MATRIX INVERSION
#    start = sample_bc.block[:,:,0]
#    transform = sample_bc.transformations[0]
#    forward = np.mod(np.dot(start,transform.matrix),2)
#    print_arrays_side_by_side(start,forward)
#    backward = np.mod(np.dot(forward,transform.get_inverted_matrix()),2)
#    print_arrays_side_by_side(forward,backward)
    #UNIT TEST FOR transform.forward and transform.backward
#    start = sample_bc.block[:,:,0]
#    transform = sample_bc.transformations[0]
#    forward = transform.forward(start)
#    print_arrays_side_by_side(start,forward)
#    backward = transform.backward(forward)
#    print_arrays_side_by_side(forward,backward)
    #UNIT TEST for block forward() and backward()
#    start = sample_bc.block[:,:,0]
#    forward = sample_bc.forward(start)
#    print_arrays_side_by_side(start,forward)
#    print(f"Probability of Bernoulli: {estimate_probability(forward)}")
#    print(f"Sum of ones: {np.sum(forward)}")
#    backward = sample_bc.backward(forward)
#    print_arrays_side_by_side(forward,backward)


# Motivations: 
#  Computational efficiency
#  Ground-up bitwise calculations
#  'Blank sheet of paper' exercise
#  Advance theory from first principles
#  Specific to Generative applications
#   reversible computing - also energy efficient

# Final architecture:

# Batch / Layer
# BatchxLayerx32x32x8
# Batch on dim 0
# Visualize as Bernoulli on dim (-1)

# Input Layer
# 0x0x32x32x8 (for batch 0)
# Certainty ==> dim(-1) is just duplicate of other dims

# Output Layer
# 0xkx32x32x8 (for batch 0, k layers)
# Label ==> dim(-1) is just duplicate of other dims (known)
# Labels are simple encoding on 32x32 grid e.g., block diagonal
# Are we just targeting supervised learning in this project? 

# Loss Function
# Expectation of Hamming Distance at Output Layer; Model vs. Label
# Monte Carlo calculation (?)

# Transforms 
# Reversible bitwise on each bit layer in 8-bit representation
# Optimization:
#   Choose 8 

# Supervised learning
# Input-2-3-4-5-6-7-Output
# Input encoding : e.g., MNIST binarized image
# Output encoding ; e.g., block diagonal .. but could be anything
#  (opportunity for modeler to incorporate prior knowledge)
# # of transforms: 7
# < Python class Bitbox that meets these criteria >
#   -has 32x32x8 grid of 1,0
#   -can evaluate Bernoulli, show as float or as sample
#   -can set input layer, output layer (model), output layer (label)  
#   -can calculate Hamming distance
#   -has 7 distinct transformations
#   -can apply its transformations in forward direction
#   -can apply its transformations in the backward direction
# Basic idea: shape layers 2-7 such that result is closer to output
#  (so Bernoulli eval gives you the Output instead of the input)
# Game: start w/ input, need to transform it to match output
#    using only reversible computations 
# Optimization : loss = hamming distance model vs. label
# Optimization algo: Monte Carlo / greedy by layer (?)
# Optimization algo: Reinforcement Learning (?) 
# Transforms act on 32x32 bit layer
# Transforms must be reversible and bit-wise
# Transforms can be convolution-like or mixing - as long as they are reversible
# < Python class Transform that meets these criteria > 
# Apply transforms in forward direction - but they're reversible
# Model evaluation : Bernoulli sample 
# probably define a fixed set of "off the shelf" transformations
# (reversible automata; CNOT; GF(2) mixture; etc.)
# then it's just a matter of sequencing 7 to fit the data

# Mixture of Models + Softmax == more complex models


# Transformation:
# 







# 32x32 grid
# traverse grid w/ 2^5 filters
# result = 2^5 grids of 31x31
# cosolidate into one grid where cell >= 16
# traverse grid w/ 2^5 filters
# etc.
# decode:
# start w/ 4x4 grid
# predict boundaries under all 2^5 filters
# consolidate boundaries where cell >= 16
# flaws:
# -> probabilities should vary depending on what we're generating
# -> locality is strictly defined in model (no attention)

# or

# time-evolution 
# rely on reversible processes as much as possible
# when you can, discard "noise"
# i.e., you are looking for situations when 'boundary' can be characterized as noise
# 
# two types of cells: determined; probable
# probable cells have priors - could be any other cell
#   (probable or determined) from prior timestep
#   must form DAG across all timesteps
# learning: start with big, complex DAG
#   and whittle / reinforce according to training data
# whole trick: learning algorithm

'''
Simulated Annealing (SA):
Principle: Inspired by the annealing process in metallurgy where a material is heated and then slowly cooled to reduce defects, simulated annealing is used to find an approximate global minimum of an objective function.
Procedure: The algorithm starts with a random solution and then, at each step, selects a neighboring solution. If the new solution is better, it is always accepted. If it's worse, it might still be accepted with a certain probability, especially in the early stages. This probability decreases as the "temperature" of the system decreases (hence the annealing analogy). The idea is to allow the algorithm to escape local minima early on when the temperature is high and then fine-tune the search as the temperature drops.
Usage: SA is a general-purpose optimization method that can be applied to many problems. It's especially useful when the search space is discrete and for combinatorial optimization problems.
'''



'''
A deep generative network is a type of neural network designed to generate new data samples that resemble a given set of training samples. The major components of a functioning deep generative network include:

1. **Network Architecture**:
   - **Layers**: Deep generative networks typically consist of multiple layers, which can be either fully connected, convolutional, recurrent, or a mix, depending on the data modality (e.g., images, sequences, etc.).
   - **Activation Functions**: Non-linear activation functions, such as ReLU (Rectified Linear Unit), tanh, or sigmoid, are used in the neurons to introduce non-linearity to the model.

2. **Latent Space**:
   - The latent space is a compressed representation from which the network can generate data. It's a lower-dimensional space where each point can be decoded to produce a data sample.

3. **Loss Function**:
   - This determines how well the generated data matches the true data distribution. The choice of loss function depends on the type of generative network. For instance, Generative Adversarial Networks (GANs) use a game-theoretic loss, while Variational Autoencoders (VAEs) use a combination of reconstruction loss and a KL-divergence term.

4. **Regularization Techniques**:
   - Techniques like dropout, batch normalization, or spectral normalization may be used to prevent overfitting and stabilize training.

5. **Optimization Algorithm**:
   - Algorithms like stochastic gradient descent (SGD), Adam, or RMSprop are used to adjust the network's weights based on the loss function.

6. **Sampling Mechanism**:
   - For generating new data points, a mechanism to sample from the latent space is needed. This could be random sampling, interpolation between points, or other methods.

7. **Network Components Specific to Certain Models**:
   - **Encoder**: In architectures like VAEs, an encoder is used to project input data into the latent space.
   - **Decoder**: This component takes points from the latent space and decodes them into data samples.
   - **Discriminator**: In GANs, the discriminator's role is to distinguish between real and generated samples.
   - **Generator**: In GANs, the generator tries to produce samples that the discriminator cannot distinguish from real samples.

8. **Training Data**:
   - A dataset representative of the desired distribution is essential. The quality and quantity of training data directly impact the performance of the generative network.

9. **Evaluation Metrics**:
   - Although inherently difficult for generative models, metrics like Inception Score (IS), Frechet Inception Distance (FID), or precision and recall scores can provide some quantitative assessment of the quality and diversity of generated samples.

10. **Regular Updates and Feedback Loops**:
   - Generative networks, especially GANs, can be notoriously hard to train, often requiring careful tuning, regular monitoring, and feedback adjustments.

11. **Initialization**:
   - Proper weight initialization can significantly impact the training dynamics and the success of the training process.

These are the major components, but the specifics can vary based on the problem, the dataset, and the chosen architecture. Moreover, training deep generative models can be tricky, often requiring expert knowledge and experience to fine-tune and stabilize.
'''
