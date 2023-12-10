import matplotlib.pyplot as plt
import numba
import numpy as np

INT_TYPE = np.int32

@numba.njit
def invert_lower_triangular(matrix):
    n = matrix.shape[0]
    inv_matrix = np.eye(n, dtype=np.int32)  # Start with the identity matrix
    for i in range(1, n):
        for j in range(i):
            inv_matrix[i, j] = np.sum(np.multiply(matrix[i, j:i], inv_matrix[j:i, j])) % 2
    return inv_matrix

@numba.njit
def invert_upper_triangular(matrix):
    n = matrix.shape[0]
    inv_matrix = np.eye(n, dtype=np.int32)  # Start with the identity matrix
    for i in range(n-2,-1,-1):
#    for i in reversed(range(n - 1)):
        for j in range(i + 1, n):
            inv_matrix[i, j] = np.sum(np.multiply(matrix[i, i + 1:j + 1], inv_matrix[i + 1:j + 1, j])) % 2
    return inv_matrix

@numba.njit
def xd_transpose(arr):
    if arr.ndim == 2:
        return np.transpose(arr)
    elif arr.ndim == 3:
        return np.transpose(arr, (0, 2, 1))
    else:
        raise ValueError("Array must be either 2D or 3D")

def print_arrays_side_by_side(array1, array2, character_mode=True):
    '''Utility function displays two 32x32 binary arrays'''
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

def plot_digit_grid(samples, title, digit_size=(32, 32)):
    fig, axes = plt.subplots(10, 10, figsize=(10, 10))
    fig.suptitle(title)
    for i, ax in enumerate(axes.flat):
        # Assuming the samples are flattened, reshape them to the original image shape
        ax.imshow(samples[i].reshape(digit_size), cmap='gray')
        ax.axis('off')
    plt.show()

def log_loss(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15) #avoid log(0)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

'''The functions below are not deployed in the final training loop setup'''

@numba.njit
def hamming_distance(a: np.array,b: np.array):
    return np.sum(a != b)

def square_target(batch_size,n=32):
    target_block = np.zeros((batch_size,n,n),dtype=INT_TYPE)
    target_block[:,12:20,12:20] = 1
    return target_block

def batch_hamming_target(input_block, training_labels):
    hamming_target = np.zeros_like(input_block,dtype=INT_TYPE)
# Same for each digit:
#    hamming_target[:,12:20,12:20] = 1
# Different for each digit, one example:
    for i in range(len(training_labels)):
        if training_labels[i] == 6:
            hamming_target[i,0:16,0:16] = 1
        if training_labels[i] == 7:
            hamming_target[i,16:32,16:32] = 1
    return hamming_target

@numba.njit
def estimate_batch_rle_complexity(arr):
    flat_batch = arr.ravel()
    changes = np.diff(flat_batch) != 0
    rle_complexity = np.sum(changes, dtype=INT_TYPE)
    return rle_complexity

@numba.njit
def bernoulli_distance(a: np.array):
    return np.abs(512-np.sum(a))


