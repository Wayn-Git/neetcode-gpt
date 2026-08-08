import numpy as np
from numpy.typing import NDArray


class Solution:
    def lookup(self, embeddings: NDArray[np.float64], token_ids: NDArray[np.int64]) -> NDArray[np.float64]:
        # embeddings: (vocab_size, embed_dim) matrix
        # token_ids: 1D array of integer token IDs
        # Return the embedding vectors for the given token IDs
        # return np.round(your_answer, 5)

        # In embeddings the vocab size represent the amount of words [tokens] we have and the embed dimention repesent the size of the vector representing each token if the emved dim is 128 that means each token has 128 numbers

        # A token ID is basically a number assigned to a token. It helps the neural network identify and retrieve the token's embedding. Token ID acts as a address in the embedding table 



        return np.round(embeddings[token_ids], 5)
