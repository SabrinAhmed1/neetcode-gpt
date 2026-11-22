import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        stabilized = z - np.max(z)
        exponentiated = np.exp(stabilized)
        sum_exp = np.sum(exponentiated)
        result = exponentiated / sum_exp
        return np.round(result, 4)
        
