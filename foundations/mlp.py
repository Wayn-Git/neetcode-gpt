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

        for w, b in zip(weights, biases):

                LayerInput = x 
            
                output = LinearLayer(LayerInput, w, b)

                # Layer = LinearLayer(current,w, b)
                a = ReLu(output) 

                x = a




        
                # l1 = LinearLayer(x, w, b)
                # a1 = ReLu(l1)

                # l2= LinearLayer(a1, w, b) 
                # a2 = ReLu(l2)

                # l3 = LinarLayer(a2, w, b)
                # a3 = ReLu(l3)

                # l4 = LinearLayer(a3)

        return np.round(output, 5)