import torch
import torch.nn as nn
from torchtyping import TensorType

class SingleHeadAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        # Create three linear projections (Key, Query, Value) with bias=False
        # Instantiation order matters for reproducible weights: key, query, value

        self.attention_dim = attention_dim

        self.K = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.Q = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.V = nn.Linear(embedding_dim, attention_dim, bias=False)

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # 1. Project input through K, Q, V linear layers
        # 2. Compute attention scores: (Q @ K^T) / sqrt(attention_dim)
        # 3. Apply causal mask: use torch.tril(torch.ones(...)) to build lower-triangular matrix,
        #    then masked_fill positions where mask == 0 with float('-inf')
        # 4. Apply softmax(dim=2) to masked scores
        # 5. Return (scores @ V) rounded to 4 decimal places
        
        queries = self.Q(embedded)
        key = self.K(embedded)
        values = self.V(embedded)

        key_transpose = key.transpose(1, 2) # dimension 1 to 2

        raw_attention_score = queries @ key_transpose / (self.attention_dim ** 0.5)

        mask = torch.tril(
    torch.ones(
        embedded.size(1),
        embedded.size(1),
        device=embedded.device
    )
)

        raw_attention_score = raw_attention_score.masked_fill(
    mask == 0,
    float("-inf")
)


        # softmax_layer = nn.Softmax(dim=2)


        attention = nn.Softmax(dim=2)(raw_attention_score)



        return torch.round(attention @ values, decimals=4)


