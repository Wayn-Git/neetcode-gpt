from typing import List


class Solution:
    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        tokens = list(corpus)
        output = []

        for _ in range(num_merges):
            # 1. Count adjacent pairs
            pairs = {}

            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                pairs[pair] = pairs.get(pair, 0) + 1

            # No pairs left to merge
            if not pairs:
                break

            # 2. Find most frequent pair.
            # If tied, choose lexicographically smallest pair.
            max_freq_pair = min(
                pairs.keys(),
                key=lambda pair: (-pairs[pair], pair)
            )

            # 3. Record the merge
            output.append(list(max_freq_pair))

            # 4. Merge all non-overlapping occurrences left-to-right
            merged_token = ''.join(max_freq_pair)
            new_tokens = []

            i = 0
            while i < len(tokens):
                if (
                    i < len(tokens) - 1
                    and tokens[i] == max_freq_pair[0]
                    and tokens[i + 1] == max_freq_pair[1]
                ):
                    new_tokens.append(merged_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1

            tokens = new_tokens

        return output