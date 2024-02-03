from neuranet import Tensor, where, exp

__all__ = ["relu"]


def relu(input: Tensor) -> Tensor:
    return where(input > 0, input, 0)

def sigmoid(input: Tensor) -> Tensor:
    return 1/(1 + exp(input))