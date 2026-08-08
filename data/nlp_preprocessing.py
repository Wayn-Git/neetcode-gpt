import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        combined_list = positive + negative

        splitted_list = [word for string in combined_list for word in string.split()]

        unique_sorted_list = sorted(set(splitted_list))

        # 2. Encode each sentence by replacing words with their IDs
        output = None
        word_map = {}

        for ID, word in enumerate(unique_sorted_list, start=1):
            word_map[word] = ID

        def encode_sentence(sentences):
            tensors = []
            for sentence in sentences:

                encode_id = [word_map[word] for word in sentence.split()]

                tensor_sm = torch.tensor(encode_id, dtype=torch.float)

                tensors.append(tensor_sm)
            return tensors

        output = encode_sentence(positive) + encode_sentence(negative)


        padded_dataset = nn.utils.rnn.pad_sequence(
            output, batch_first=True, padding_value=0.0
        )

        return padded_dataset
