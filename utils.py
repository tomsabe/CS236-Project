import matplotlib.pyplot as plt
import numpy as np

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

def hamming_distance_a_b(a: np.array,b: np.array):
    return np.sum(a != b)

def bernoulli_distance(a: np.array):
    return np.abs(512-np.sum(a))
