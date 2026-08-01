import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        # x: 1D input array
        # weights: list of 2D weight matrices
        # biases: list of 1D bias vectors
        # Apply ReLU after each hidden layer, no activation on output layer
        # return np.round(your_answer, 5)

        # print("weights",np.shape(weights))
        # print("bias", np.shape(biases))
        # print("x", np.shape(x))
        
        def ReLu(z):
            return np.maximum(0, z)

        def LinearLayer(data, weight, bias):
            return np.dot(data, weight) + bias

        current = x 

        for i, (w, b) in enumerate(zip(weights, biases)):

                output = LinearLayer(current, w, b)
            
                if i == len(weights) - 1:
                        current = output          # No ReLU
                else:
                        current = ReLu(output)    # Hidden layer


        return np.round(current, 5)