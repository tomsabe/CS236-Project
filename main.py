import matplotlib.pyplot as plt
import numpy as np
import random
from tqdm import tqdm
from scipy.stats import binom_test

from load_mnist import load_mnist_binarized

#for block diagnoal experiment:
#from generate_label import generate_label

def estimate_probability(sequence: np.array):
    num_ones = np.sum(sequence)
    n = 1024
    p = 0.5
    probability = binom_test(num_ones, n, p)
    return probability

def rle_length_32bit(segment):
    if len(segment) == 0:
        return 0

    compressed_length = 0
    current_bit = segment[0]
    count = 1

    for bit in segment[1:]:
        if bit == current_bit:
            count += 1
        else:
            compressed_length += 1 + len(str(count))
            current_bit = bit
            count = 1

    compressed_length += 1 + len(str(count))
    return compressed_length

def estimate_rle_length(array):
    total_compressed_length = 0
    for row in array:
        total_compressed_length += rle_length_32bit(row)
    return total_compressed_length

def print_arrays_side_by_side(array1, array2, character_mode=True):
    '''
    Utility function displays two 32x32 binary arrays
    '''
    # Ensure arrays are 2D and 32x32
    if array1.ndim == 3:
        array1 = array1.reshape(array1.shape[0], array1.shape[1])
    if array2.ndim == 3:
        array2 = array2.reshape(array2.shape[0], array2.shape[1])
    if array1.shape != (32, 32) or array2.shape != (32, 32):
        raise ValueError("Both arrays must be of shape (32, 32) or (32, 32, 1)")

    if character_mode:
        # Print arrays side by side
        for row1, row2 in zip(array1, array2):
            print("".join(str(x) for x in row1) + "    " + "".join(str(x) for x in row2))
        print("\n")
    else:
        combined_array = np.concatenate((array1, array2), axis=1)
        plt.imshow(combined_array, cmap='gray', interpolation='nearest')
        plt.axis('off')  # Turn off the axis
        plt.show()

def invert_lower_triangular(matrix):
    n = matrix.shape[0]
    inv_matrix = np.eye(n, dtype=int)  # Start with the identity matrix
    for i in range(1, n):
        for j in range(i):
            inv_matrix[i, j] = np.sum(np.multiply(matrix[i, j:i], inv_matrix[j:i, j])) % 2
    return inv_matrix

def invert_upper_triangular(matrix):
    n = matrix.shape[0]
    inv_matrix = np.eye(n, dtype=int)  # Start with the identity matrix
    for i in reversed(range(n - 1)):
        for j in range(i + 1, n):
            inv_matrix[i, j] = np.sum(np.multiply(matrix[i, i + 1:j + 1], inv_matrix[i + 1:j + 1, j])) % 2
    return inv_matrix

def hamming_distance_a_b(a: np.array,b: np.array):
    return np.sum(a != b)

def bernoulli_distance(a: np.array):
    return np.abs(512-np.sum(a))

class Transformation:
    def __init__(self, matrix=None, bias=None):
        '''
        Initializes the Transformation with a 32x32 binary matrix.
        If no matrix is provided, generates a random invertible matrix.
        '''
        # Initialize the transformation matrix
        if matrix is not None:
            if matrix.shape != (32, 32):
                raise ValueError("Matrix should be of shape (32, 32)")
            self.matrix = matrix
            self.inverted_matrix = None
            self.lower = None
            self.upper = None
        else:
            self.matrix, self.lower, self.upper = self.make_random_invertible()
            self.inverted_matrix = None
        # Initialize the bias
        if bias is not None:
            if bias.shape != (32, 32):
                raise ValueError("Bias should be of shape (32, 32)")
            self.bias = bias
        else:
            self.bias = np.random.randint(0,2,(32,32))

    def get_inverted_matrix(self) -> np.array:
        if self.inverted_matrix is not None:
            return self.inverted_matrix
        # otherwise, calculate it
        inverted_lower = invert_lower_triangular(self.lower)
        inverted_upper = invert_upper_triangular(self.upper)
        self.inverted_matrix = np.mod(np.dot(inverted_upper,inverted_lower),2)
        return self.inverted_matrix

    def make_random_invertible(self) -> tuple:
        '''
        Generate a random 32x32 invertible matrix over GF(2).
        '''
        # Lower triangular matrix with ones on diagonal
        lower = np.tril(np.random.randint(0, 2, (32, 32)))
        np.fill_diagonal(lower, 1)
        
        # Upper triangular matrix with ones on diagonal
        upper = np.triu(np.random.randint(0, 2, (32, 32)))
        np.fill_diagonal(upper, 1)
        
        # Multiply matrices over GF(2)
        invertible_matrix = np.mod(np.dot(lower, upper), 2)
        
#        print_arrays_side_by_side(lower,upper)

        return invertible_matrix, lower, upper

    def backward(self, input_grid: np.ndarray) -> np.ndarray:
        '''
        Apply the transformation in the inverse direction using GF(2) arithmetic.
        '''
        inverted_matrix = self.get_inverted_matrix()
        result = np.dot((input_grid + self.bias) % 2, inverted_matrix) % 2 
        return result

    def forward(self, input_grid: np.ndarray) -> np.ndarray:
        '''
        Apply the transformation in the forward direction using GF(2) arithmetic.
        '''
        result = (np.dot(input_grid, self.matrix) % 2 + self.bias) % 2
        return result

class BitCube:
    def __init__(self, first_layer: np.ndarray, label=None):
        '''
        Initializes the BitCube.
        
        :param first_layer: The input layer of shape (32, 32).
        :param label: Target 32x32 encoding for the output layer.
        '''
        # The "cube" (block) holds the input layer
        # plus all subsequent calculated layers
        self.cube = np.zeros((32, 32, 1), dtype=int)
        self.cube[:, :, 0] = first_layer
        # The "transformations" holds the list of transformation matrices
        self.transformations = []
        # The "label" holds the target for the transformations, or "None"
        self.label = label

    def backward(self, layer: np.array) -> np.array:
        # execute transformations in backward direction
        for i in range(1,len(self.transformations)+1):
            layer = self.transformations[-i].backward(layer)
        return layer

    def forward(self, layer: np.array) -> np.array:
        # execute transformations in forward direction
        for i in range(0,len(self.transformations)):
            layer = self.transformations[i].forward(layer)
        return layer

    def add_transformation(self, transformation: Transformation):
        '''
        Add a transformation to the BitCube and increase its depth.
        '''
        # Update list of transformations
        self.transformations.append(transformation)

        # Add the new layer to the cube
        new_layer = transformation.forward(self.cube[:, :, - 1])
        new_layer = np.reshape(new_layer, (32,32,1))
        self.cube = np.concatenate((self.cube, new_layer), axis=2)
        return self.cube

    def distance_to_label(self) -> int:
        '''
        Compute the hamming distance between the output layer and the label.
        '''
        if self.label is None:
            raise ValueError("No label set for this BitCube.")
        
        # Compute the hamming distance for the last layer
        return hamming_distance_a_b(self.cube[:, :, -1], self.label)

    def try_transformation(self, transformation):
         '''Try out the outcome of applying a transformation, but do not actually apply it.'''
         return transformation.forward(self.cube[:, :, - 1])

    def set_label(self, label_matrix):
        self.label = label_matrix

    def view_layer(self, n: int):
        '''
        Prints the nth layer as a 32x32 grid of 1's and 0's.
        '''
        if n < 0 or n >= self.cube.shape[2]:
            raise ValueError(f"Layer {n} is out of bounds.")
        
        for row in self.cube[:, :, n]:
            print(' '.join(map(str, row)))

    def render_cube(self, layer_1: int, layer_2: int) -> np.ndarray:
        '''
        Render a 2D visualization of the cube based on summed bits from layer_1 to layer_2.
        
        :return: A 32x32 grid generated based on Bernoulli probabilities.
        '''
        if layer_1 < 0 or layer_2 < 0 or layer_1 >= self.cube.shape[2] or layer_2 >= self.cube.shape[2] or layer_1 > layer_2:
            raise ValueError(f"Invalid layer range: {layer_1} to {layer_2}")

        summed_grid = np.sum(self.cube[:, :, layer_1:layer_2+1], axis=2)
        num_layers = layer_2 - layer_1 + 1
        probability_grid = summed_grid / num_layers

        # Generate a new 32x32 grid based on the Bernoulli probabilities
        bernoulli_draw = np.random.binomial(n=1, p=probability_grid)
        
        return bernoulli_draw

class BitCubeBatch:
    def __init__(self, batch_size=512):
        self.batch_size = batch_size
        # Load binarized MNIST dataset
        self.train_images, self.train_labels = load_mnist_binarized()
        self.batch = []

    def get_batch(self, target_label=0):
        '''Get a batch of binarized MNIST digits.'''      
        # Randomly select indices for the batch
        indices = np.random.choice(len(self.train_images), self.batch_size, replace=False)
        batch_images = self.train_images[indices]
        batch_labels = self.train_labels[indices]      
        # Convert images and labels to BitCubes
        for img, label in zip(batch_images, batch_labels):
            if label == target_label:
                label_matrix = np.ones((32,32,1), dtype=int)
            else:
                label_matrix = np.zeros((32, 32, 1), dtype=int)
            bitcube = BitCube(img)
            bitcube.set_label(label_matrix)
            self.batch.append(bitcube)
        return self.batch
   
    def get_sample_cube(self):
        return random.choice(self.batch)

class GreedyOptimizer:
    def __init__(self, batch_size=512):
        self.bitcube_batch = BitCubeBatch(batch_size=batch_size)

    def optimize(self, iterations=1000, display_every=10):
        # Get a batch
        batch = self.bitcube_batch.get_batch()

        #count layers
        layers = 1

        # Calculate the initial distance
        total_distance = sum([bernoulli_distance(bc.cube[:,:,-1]) for bc in batch]) \
            -sum([estimate_rle_length(bc.cube[:,:,-1]) for bc in batch])

        for iteration in tqdm(range(iterations)):
            # Get a new random invertible transformation
            new_transform = Transformation()
            new_transform.make_random_invertible()
            new_distance = sum([bernoulli_distance(bc.try_transformation(new_transform)) for bc in batch]) \
                -sum([estimate_rle_length(bc.try_transformation(new_transform)) for bc in batch])

            # If the new distance is smaller, add the layer
            if new_distance < total_distance:
                for bc in batch:
                    bc.add_transformation(new_transform) #actually we only need one transformation flow
                layers += 1
                total_distance = new_distance

            # Display progress
            if iteration % display_every == 0:
                sample_batch = random.sample(batch,30)
                p_bern = sum([estimate_probability(sample.cube[:,:,-1]) for sample in sample_batch])/30
                rle_length = sum([estimate_rle_length(sample.cube[:,:,-1]) for sample in sample_batch])/30
                print(f"Iteration: {iteration} | Total Distance: {total_distance} | Layers: {layers} | p(Bern): {p_bern} | RLE length: {rle_length}")
#                print("Samples of transformed matrices:")
#                bc_samples = random.sample(batch,2)
#                print_arrays_side_by_side(bc_samples[0].cube[:,:,-1],bc_samples[1].cube[:,:,-1])
#                input("Press any key to continue...")
               
# Example of usage:
if __name__ == "__main__":
    optimizer = GreedyOptimizer(batch_size=100)
    optimizer.optimize(iterations=4000, display_every=1000)
    #take a sample from the batch
    sample_bc = optimizer.bitcube_batch.get_sample_cube()
    #1 print first and last layers
    print_arrays_side_by_side(sample_bc.cube[:,:,0],sample_bc.cube[:,:,-1],character_mode=False)
    #2 now start with the final layer and show the inverse
    start = sample_bc.cube[:,:,-1]
    backward = sample_bc.backward(start)
    print_arrays_side_by_side(start,backward,character_mode=False)
    #3 a 'noised' input
    noisy_start = start ^ np.random.choice([0, 1], size=start.shape, p=[0.99, 0.01])
    backward = sample_bc.backward(noisy_start)
    print_arrays_side_by_side(start,backward,character_mode=False)
    #4 finally, a totally random input
    start = np.random.randint(0,2,(32,32))
    backward = sample_bc.backward(start)
    print_arrays_side_by_side(start,backward,character_mode=False)


    #UNIT TEST FOR THE MATRIX INVERSION
#    start = sample_bc.cube[:,:,0]
#    transform = sample_bc.transformations[0]
#    forward = np.mod(np.dot(start,transform.matrix),2)
#    print_arrays_side_by_side(start,forward)
#    backward = np.mod(np.dot(forward,transform.get_inverted_matrix()),2)
#    print_arrays_side_by_side(forward,backward)
    #UNIT TEST FOR transform.forward and transform.backward
#    start = sample_bc.cube[:,:,0]
#    transform = sample_bc.transformations[0]
#    forward = transform.forward(start)
#    print_arrays_side_by_side(start,forward)
#    backward = transform.backward(forward)
#    print_arrays_side_by_side(forward,backward)
    #UNIT TEST for cube forward() and backward()
#    start = sample_bc.cube[:,:,0]
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
