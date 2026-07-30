import numpy as np
from typing import List


class Solution:
    def rms_norm(self, x: List[float], gamma: List[float], eps: float) -> List[float]:
        # Implement RMS Normalization (similar to LayerNorm but without mean centering or beta)
        # Normalize x, then scale by gamma
        # Return result rounded to 4 decimal places as a list

        e = pow(10 , -5)


        sqr_x = np.square(x)
        print(sqr_x)
        mean_x = np.mean(sqr_x)
        print(mean_x)
        eps = mean_x + e
        print(eps)
        RMS_denominaotor= np.sqrt(eps)
        print(sqr_x)
        normalized_vector = x/RMS_denominaotor
        output = normalized_vector * gamma

        print(output)

        return np.round(output, 4)