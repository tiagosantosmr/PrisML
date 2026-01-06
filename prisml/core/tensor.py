import numpy as np
from typing import Union, Tuple

class Tensor:
    """
    The Tensor is the fundamental data structure in PrisML.

    Essentially, it is a multi-dimensional array. In this implementation, it is a wrapper around a NumPy array.

    It can be created from a NumPy array, a list, an integer or a float.

    We take advantage of NumPy's broadcasting to allow for element-wise operations between Tensors of different shapes.
    But note that broadcasting only works if the shapes are compatible (aligning right to left).

    The Tensor class also supports automatic differentiation.
    If requires_grad=True, the Tensor builds a computational graph (history of operations) which later
    allows backpropagation to compute gradients.
    This allows for backpropagation, where we can automatically compute gradients.

    self.data is the actual numerical values of the Tensor.
    self.grad is the accumulated gradient of the loss with respect to this Tensor. (dLoss/dThisTensor).
        Used by optimizers to update self.data.

    """
    def __init__(self, data: Union[np.ndarray, list, int, float], requires_grad: bool = False):
        self.requires_grad = requires_grad
        self.grad = None #TODO
        if isinstance(data, Tensor):
            self.data = data.data.astype(np.float32, copy=True)
        else:
            self.data = np.array(data, dtype=np.float32)

    def __repr__(self): # print(t1)
        return f"Tensor data={self.data} requires_grad={self.requires_grad}"

    def __add__(self, other): # t1 + t2
        if isinstance(other, Tensor):
            return Tensor(self.data + other.data) #NumPy automatically broadcasts compatible shapes
        return Tensor(self.data + other)

    def __sub__(self, other): # t1 - t2
        if isinstance(other, Tensor):
            return Tensor(self.data - other.data)
        return Tensor(self.data - other)

    def __mul__(self, other): # t1 * t2
        if isinstance(other, Tensor):
            return Tensor(self.data * other.data)
        return Tensor(self.data * other)

    def __truediv__(self, other): # t1 / t2
        if isinstance(other, Tensor):
            return Tensor(self.data / other.data)
        return Tensor(self.data / other)

    @property
    def shape(self):
        return self.data.shape
        