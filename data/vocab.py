from typing import Dict, List, Tuple

class Solution:
    def build_vocab(self, text: str) -> Tuple[Dict[str, int], Dict[int, str]]:
        # Return (stoi, itos) where:
        # - stoi maps each unique character to a unique integer (sorted alphabetically)
        # - itos is the reverse mapping (integer to character)

        unique_char = sorted(set(text))

        stoi = {}
        itos = {}

        for Id, char in enumerate(unique_char):
            stoi[char] = Id

        for Id, char in enumerate(unique_char):
            itos[Id] = char

        



        return (stoi, itos)

    def encode(self, text: str, stoi: Dict[str, int]) -> List[int]:
        # Convert a string to a list of integers using stoi mapping

        nums = []

        for char in text:
            nums.append(stoi[char])

        return nums

    def decode(self, ids: List[int], itos: Dict[int, str]) -> str:
        # Convert a list of integers back to a string using itos mapping
        text = ""

        for num in ids:
            text += itos[num]


        return text
