from typing import Tuple, Any
from . import Tensor
import numpy as np 


__all__ = ["dot", "rand", "zeros", "multiply" ,"tensor2strings", "sum"
           , "where", "norm", "exp"]


def dot(tensor1: Tensor, tensor2: Tensor) -> Tensor:
    """
        This static method defines the dot product.

        Args:
            tensor1: the first Tensor.
            tensor2: the second Tensor.

        Returns:
            A Tensor, the result of the dot product of tensor1 and tensor2.
    """

    # if tensor1.shape[1] != tensor2.shape[0]:
    #     s1, s2 = tensor1.shape, tensor2.shape
    #     raise ValueError(f": shapes {s1} and {s2} not aligned: {s1[1]} != {s2[0]}")
    
    return Tensor(np.dot(tensor1, tensor2))


def rand(*shape: Tuple[int]) -> Tensor:
    """
        This static method creates a Tensor with random values.

        Args:
            shape: a tuple of integers, defining the shape of the Tensor.

        Returns:
            A Tensor.
    """

    return Tensor(np.random.rand(*shape))



def zeros(*shape: Tuple[int]) -> Tensor:
    """
        This static method creates a Tensor of zeros.

        Args:
            shape: a tuple of integers, defining the shape of the Tensor.

        Returns:
            A Tensor.
    """

    return Tensor(np.zeros(shape))


def multiply(tensor1: Tensor, tensor2: Tensor) -> Tensor:
    #TODO: write docstrings for multiply:
     
    return Tensor(np.multiply(tensor1, tensor2))


def sum(*args, **kwargs) -> Tensor:
    return Tensor(np.sum(*args, **kwargs))

def norm(input: Tensor) -> Tensor:
    return Tensor(np.array(np.linalg.norm(input)))


def where(tensor: Tensor[bool], val1: Any, val2: Any):
    return Tensor(np.where(tensor, val1, val2))

def exp(tensor: Tensor) -> Tensor:
    return Tensor(np.exp(tensor))


def tensor2strings(tensor: Tensor) -> str:
    return np.array2string(tensor)