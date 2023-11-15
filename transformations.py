import numpy as np

from utils import invert_lower_triangular, invert_upper_triangular

DEFAULT_SIZE=32
ALL_PAIRS= [(i, j) for i in range(DEFAULT_SIZE) for j in range(DEFAULT_SIZE)]

class Transformation:
    def __init__(self):
        pass

    def forward(self, input_grid: np.ndarray) -> np.ndarray:
        return input_grid #by default, identity

    def backward(self, input_grid: np.ndarray) -> np.ndarray:
        return input_grid #by default, identify

class GF2LinearLayer(Transformation):
    def __init__(self, n=32):
        self.matrix = np.eye(n, dtype=int)
        self.inverted_matrix_cache = None
        self.lower = np.eye(n, dtype=int)
        self.upper = np.eye(n, dtype=int)
        self.bias = np.zeros((n,n), dtype=int)

    def inverted_matrix(self) -> np.array:
        '''Use this method to access the inverted matrix'''
        if self.inverted_matrix_cache is None:
            inverted_lower = invert_lower_triangular(self.lower)
            inverted_upper = invert_upper_triangular(self.upper)
            self.inverted_matrix_cache = np.mod(np.dot(inverted_upper,inverted_lower),2)
        return self.inverted_matrix_cache

    def forward(self, input_grid: np.ndarray) -> np.ndarray:
        '''Apply the forward transformation'''
        return np.transpose((np.dot(input_grid, self.matrix) % 2 + self.bias) % 2)
#        return (np.dot(input_grid, self.matrix) % 2 + self.bias) % 2

    def backward(self, input_grid: np.ndarray) -> np.ndarray:
        '''Apply the inverse transformation.'''
        inverted_matrix = self.inverted_matrix()
        return np.dot((np.transpose(input_grid) + self.bias) % 2, inverted_matrix) % 2
#        return np.dot((input_grid + self.bias) % 2, inverted_matrix) % 2

    @classmethod
    def random_invertible(cls, n=DEFAULT_SIZE, num_tri_ones=1, num_bias_ones=1):
        '''Create a random GF2 Linear Layer'''
        # Initialize lower and upper triangular matrices
        lower = np.zeros((n, n), dtype=int)
        upper = np.zeros((n, n), dtype=int)
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
        bias = np.zeros((n,n), dtype=int)
        selected_indices = np.random.choice(len(ALL_PAIRS), num_bias_ones, replace=False)
        for index in selected_indices:
            x, y = ALL_PAIRS[index]
            bias[x, y] = 1
        # Create and return the instance
        instance=cls(n)
        instance.lower = lower
        instance.upper = upper
        instance.matrix= invertible_matrix
        instance.bias = bias
        return instance


if __name__ == '__main__':
    # Unit tests for GF2LinearLayer

    # Test 1: Ensure that the forward and backward transformations are inverses
    layer = GF2LinearLayer.random_invertible()
    input_grid = np.random.randint(0, 2, (32, 32))

    forward_result = layer.forward(input_grid)
    backward_result = layer.backward(forward_result)

    assert np.array_equal(backward_result, input_grid), "Backward transformation did not correctly invert the forward transformation"

    # Test 2: Identity transformation test
    identity_layer = GF2LinearLayer()
    identity_result = identity_layer.forward(input_grid)
    assert np.array_equal(identity_result, input_grid), "Identity transformation failed"

    print("All tests passed.")
