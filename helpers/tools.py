import numpy as np
from torch import Tensor


def channel_loss(input_data: Tensor, output_data: Tensor, error: str = 'MSE'):
    """
    Measure generator's loss in replicating input data
    """

    assert input_data.shape == output_data.shape

    if error == 'MSE':
        difference = output_data - input_data
        loss = np.square(difference).mean()
    else:
        loss = -1
    return loss.item()