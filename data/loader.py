import torch
from torchtyping import TensorType
from typing import Tuple

class Solution:
    def create_batches(self, data: TensorType[int], context_length: int, batch_size: int) -> Tuple[TensorType[int], TensorType[int]]:
        # data: 1D tensor of encoded text (integer token IDs)
        # context_length: number of tokens in each training example
        # batch_size: number of examples per batch
        #
        # Return (X, Y) where:
        # - X has shape (batch_size, context_length)
        # - Y has shape (batch_size, context_length)
        # - Y is X shifted right by 1 (Y[i][j] = data[start_i + j + 1])
        #
        # Use torch.manual_seed(0) before generating random start indices
        # Use torch.randint to pick random starting positions
        torch.manual_seed(0)

        start = torch.randint(len(data)-context_length, size=(batch_size, )) # Subtracting the context_lenght to reserve the space for  as y = technique a index further x. If we do the complete window it would have a problem like this

        """
        start=6
        X = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        X = 70, 80, 90
        Y = 80, 90, ?

        generates:

        low ≤ random number < high
        so low and never be high

        """
        X = []
        Y = []
        

            
        for j in start:
            X.append(data[j: j + context_length])
              
            Y.append(data[j+1 : j+context_length + 1])
        print(X, Y)

        X = torch.stack(X)
        Y = torch.stack(Y)

        
        return (X, Y)
