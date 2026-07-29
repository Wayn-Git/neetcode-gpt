import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)

        # x
        # linear1 (l1)
        # relu1 (a1)
        # linear2 (l2)
        # prediction (l2)
        # Loss (MSE)
        # gradient (W1)
        # chain rule  



        #  the ReLU derivative is a binary mask. -> Gradient

        # Defining the activation function

        def ReLu(z):
            return np.maximum(0, z)

        def relu_derivative(u):
            return (u >0).astype(int)


        

    
      

  
        iterations = 100


        l1 = np.dot(W1, x) + b1

        a1 = ReLu(l1)

        l2 = np.dot(W2, a1) + b2

        MSE = np.mean(np.square(l2-y_true))

        print(l1)


        N = len(l2)

        # Calculating the derivatives of our functions (l1, a1, l2, MSE)

        dW1_l1 = x

        dReLu_l1 = relu_derivative(l1)
        
        db1 = 1

        dl2 = W2

        # print("Relu: ",dRelu)

        delta2 = 2 * (l2 - y_true) / N



        dW2 = np.outer(delta2, a1)
        db2 = delta2

        # dW1 = delta2 * dl2 * dReLu_l1 * dW1_l1
        delta1 = np.dot(delta2, dl2) * dReLu_l1
        dW1 = np.outer(delta1, dW1_l1)
        db1 = delta1 
        print(MSE)

        output = {
            'loss': np.round(MSE, 4),
            'dW1': np.round(dW1, 4),
            'db1': np.round(db1, 4),
            'dW2': np.round(dW2, 4),
            'db2': np.round(db2,4)
        }
        
        return output


