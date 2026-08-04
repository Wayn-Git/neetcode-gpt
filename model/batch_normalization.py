import numpy as np
from typing import Tuple, List


class Solution:
    def batch_norm(self, x: List[List[float]], gamma: List[float], beta: List[float],
                   running_mean: List[float], running_var: List[float],
                   momentum: float, eps: float, training: bool) -> Tuple[List[List[float]], List[float], List[float]]:
        # During training: normalize using batch statistics, then update running stats
        # During inference: normalize using running stats (no batch stats needed)
        # Apply affine transform: y = gamma * x_hat + beta
        # Return (y, running_mean, running_var), all rounded to 4 decimals as lists


        #------ Converting the normal lists to float so it's reliable to compute ------

        x = np.array(x)
        gamma = np.array(gamma)
        beta = np.array(beta)
        running_mean = np.array(running_mean)
        running_var = np.array(running_var)

# ---------------------------------------------------------

        # ------ Calculating the mean of the feature and varience of the batch -----

        feature_mean = np.mean(x,axis=0)
        varience_batch = np.var(x,axis=0)

# ---------------------------------------------------------



        if training == True:

        # ----- Normalazing the input features -----

           x_f_t = (x - feature_mean)/np.sqrt(varience_batch + eps)

        # ---- Calculating the final output ----

           y = x_f_t * gamma + beta 

# ---------------------------------------------------------


        # ----- Updating the mean and the varience -----

           running_mean = (1-momentum) * running_mean + momentum * feature_mean
           running_var = (1-momentum) * running_var + momentum * varience_batch

# ---------------------------------------------------------


        
        elif training == False:

# ------ Normalizing the features with the updated mean and the varience as we're no longer training -----

            x_f_i = (x -running_mean)/np.sqrt(running_var + eps)

# ----- Gamma and beta are learnable parameters that let the neural network adjust the normalized output. -----

            y = gamma * x_f_i + beta
        
# ---------------------------------------------------------



        y = np.round(y,4)
        r_m = np.round(running_mean,4)
        r_v = np.round(running_var,4)


        y = y.tolist()
        r_m = r_m.tolist()
        r_v = r_v.tolist()

        output = tuple([y, r_m, r_v])

        
        
        return output
