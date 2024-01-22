from abc import ABC, abstractmethod
from ...nn import Layer, Module
from ... import Tensor, sum
from neuranet import Tensor
from typing import List, Any, Self

__all__ = ["Loss", "MSELoss"]

class Loss(Module):
    
    out_grad: Tensor = None 

    def __init__(self, layers: List[Layer]) -> None:
        super(Loss, self).__init__()
        self.layers: List[Layer] = layers

    @abstractmethod
    def forward(self, *args, **kwargs) -> Self:
        ...

    def backward(self) -> None:
        grad = self.out_grad
        for layer in reversed(self.layers):
            grad = layer.backward(grad)



class MSELoss(Loss):
    
    def __init__(self, layers: List[Layer]) ->None:
        super(MSELoss, self).__init__(layers)
    
    def forward(self, y_pred: Tensor, y: Tensor) -> Loss:
        self.out_grad = (2/y.shape[0])*Tensor((y_pred - y)) 
        return self