from neuranet.nn.base import Activation
import neuranet.nn.functional as F 

__all__ = ["ReLU", "Sigmoid"]


class ReLU(Activation):
    
    def __init__(self) -> None:

        super(ReLU, self).__init__(
            active_func = F.relu,
            active_prime = F.relu
        )

class Sigmoid(Activation):

    def __init__(self) -> None:

        super(Sigmoid, self).__init__(
            active_func = F.sigmoid,
            active_prime = lambda input: F.sigmoid(input)*(1 - F.sigmoid(input))
        )