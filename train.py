import torch
import torch.nn as nn
import torch.nn.functional as F

# The GPT model is provided for you. It returns raw logits (not probabilities).
# You only need to implement the training loop below.

class Solution:
    def train(self, model: nn.Module, data: torch.Tensor, epochs: int, context_length: int, batch_size: int, lr: float) -> float:
        # Train the GPT model using AdamW and cross_entropy loss.
        # For each epoch: seed with torch.manual_seed(epoch),
        # sample batches from data, run forward/backward, update weights.
        # Return the final loss rounded to 4 decimals.
        

        data_len = len(data)

        high = data_len - context_length

        
# The - length so that we don't end up with missing values  
        


        

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        

        for epoch in range(epochs):

            torch.manual_seed(epoch)

            X = []
            Y = []

            

            indicies = torch.randint(0, high, (batch_size,))

            for i in indicies:
                X.append(data[i:i+context_length])
                Y.append(data[i+1: i+context_length+1])

            X = torch.stack(X)
            Y = torch.stack(Y) 

            logits = model(X)
            logits_flat = torch.flatten(logits, start_dim=0, end_dim=1)  # (B*T, C)

            Y_flat = torch.flatten(Y)  # (B*T)

            loss = F.cross_entropy(logits_flat, Y_flat)

            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return round(loss.item(), 4)

            








        


