

import numpy as np
import pickle

from utils import invert_lower_triangular, invert_upper_triangular, xd_transpose
from utils import INT_TYPE

DEFAULT_SIZE=32
ALL_PAIRS= [(i, j) for i in range(DEFAULT_SIZE) for j in range(DEFAULT_SIZE)]

class Transformation:
    def __init__(self):
        pass

    def forward(self, input_grid: np.ndarray) -> np.ndarray:
        return input_grid #by default, identity

    def backward(self, input_grid: np.ndarray) -> np.ndarray:
        return input_grid #by default, identify

class TransformationFlow(Transformation):
    def __init__(self):
        self.transformations = []
        self.latent_distribution = None
    
    def append(self, transformation: Transformation):
        self.transformations.append(transformation)

    def forward(self, input_grid: np.ndarray) -> np.ndarray:
        result_grid = input_grid
        for transformation in self.transformations:
            result_grid = transformation.forward(result_grid)
        return result_grid

    def backward(self, input_grid: np.ndarray) -> np.ndarray:
        result_grid = input_grid
        for transformation in self.transformations[::-1]:
            result_grid = transformation.backward(result_grid)
        return result_grid

    def num_layers(self) -> int:
        return len(self.transformations)

    def parameter_count(self):
        layers = self.num_layers()
        bits = layers * 32 * 32
        float_32s = bits/32
        latent_space = DEFAULT_SIZE*DEFAULT_SIZE
        params = float_32s + latent_space
        print(f"Layers: {layers}\tBits: {bits}\tFloat32 Equivalents: {float_32s}")
        print(f"Latent Space Probabilities: {latent_space}")
        print(f"Total float32 equivalents: {params} ")
        return params

    def save(self, file_name: str):
        with open(file_name,"wb") as file:
            pickle.dump(self,file)

    @classmethod
    def load(cls, file_name: str):
        with open(file_name,"rb") as file:
            return pickle.load(file)

class GF2LinearLayer(Transformation):
    def __init__(self, n=32, transpose=True):
        self.matrix = np.eye(n, dtype=np.int32)
        self.inverted_matrix_cache = None
        self.lower = np.eye(n, dtype=np.int32)
        self.upper = np.eye(n, dtype=np.int32)
        self.bias = np.zeros((n,n), dtype=np.int32)
        self.transpose = transpose

    def inverted_matrix(self) -> np.array:
        '''Use this method to access the inverted matrix'''
        if self.inverted_matrix_cache is None:
            inverted_lower = invert_lower_triangular(self.lower)
            inverted_upper = invert_upper_triangular(self.upper)
            self.inverted_matrix_cache = np.mod(np.dot(inverted_upper,inverted_lower),2)
        return self.inverted_matrix_cache

    def forward(self, input_grid: np.ndarray) -> np.ndarray:
        '''Apply the forward transformation'''
        if self.transpose:
            return xd_transpose((np.dot(input_grid, self.matrix) % 2 + self.bias) % 2)
        return (np.dot(input_grid, self.matrix) % 2 + self.bias) % 2

    def backward(self, input_grid: np.ndarray) -> np.ndarray:
        '''Apply the inverse transformation.'''
        inverted_matrix = self.inverted_matrix()
        if self.transpose:
            return np.dot((xd_transpose(input_grid) + self.bias) % 2, inverted_matrix) % 2
        return np.dot((input_grid + self.bias) % 2, inverted_matrix) % 2

    @classmethod
    def bias_only(cls, bias: np.array):
        '''Create a GF2 Linear Layer that only applies the given Bias'''
        n = bias.shape[0]
        if n != bias.shape[1]:
            raise ValueError("Bias must be a square matrix")
        instance = cls(n=n, transpose=False)
        instance.bias = bias
        return instance

    @classmethod
    def random_invertible(cls, n=DEFAULT_SIZE, num_tri_ones=1, num_bias_ones=1, transpose=True):
        '''Create a random GF2 Linear Layer'''
        # Initialize lower and upper triangular matrices
        lower = np.zeros((n, n), dtype=np.int32)
        upper = np.zeros((n, n), dtype=np.int32)
        selected_indices = np.random.choice(len(ALL_PAIRS), num_tri_ones, replace=False)
        for index in selected_indices:
            x, y = ALL_PAIRS[index]
            if x < y:
                upper[x, y] = 1
            if x > y:
                lower[x, y] = 1
        # Set all diagonal elements to 1
        np.fill_diagonal(lower, 1)
        np.fill_diagonal(upper, 1)        
        # Multiply matrices over GF(2)
        invertible_matrix = np.mod(np.dot(lower, upper), 2)
        # Now set the bias matrix
        bias = np.zeros((n,n), dtype=np.int32)
        selected_indices = np.random.choice(len(ALL_PAIRS), num_bias_ones, replace=False)
        for index in selected_indices:
            x, y = ALL_PAIRS[index]
            bias[x, y] = 1
        # Create and return the instance
        instance=cls(n=n, transpose=transpose)
        instance.lower = lower
        instance.upper = upper
        instance.matrix= invertible_matrix
        instance.bias = bias
        return instance

if __name__ == '__main__':
    # Unit tests for GF2LinearLayer

    # Test 1: Ensure that the forward and backward transformations are inverses
    layer = GF2LinearLayer.random_invertible()
    input_grid = np.random.randint(0, 2, (32, 32), dtype=INT_TYPE)

    forward_result = layer.forward(input_grid)
    backward_result = layer.backward(forward_result)

    assert np.array_equal(backward_result, input_grid), "Backward transformation did not correctly invert the forward transformation"

    # Test 2: Identity transformation test
    identity_layer = GF2LinearLayer(transpose=False)
    identity_result = identity_layer.forward(input_grid)
    assert np.array_equal(identity_result, input_grid), "Identity transformation failed"

    print("All tests passed.")

    # Test 3: Bias-Only Test
    bias = np.random.randint(0, 2, (32, 32), dtype=INT_TYPE)
    bias_layer = GF2LinearLayer.bias_only(bias=bias)
    input_grid = np.random.randint(0, 2, (32, 32), dtype=INT_TYPE)
    print(f"bias input sum: {np.sum(input_grid)}")
    forward_result = layer.forward(input_grid)
    print(f"biased sum: {np.sum(forward_result)}")
    backward_result = layer.backward(forward_result)
    print(f"reversed bias sum: {np.sum(backward_result)}")
    assert np.array_equal(backward_result, input_grid), "Identity transformation failed"

