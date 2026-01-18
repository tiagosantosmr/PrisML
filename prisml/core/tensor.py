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

    #TODO: self.grad, not implemented yet
    self.data is the actual numerical values of the Tensor.
    self.grad is the accumulated gradient of the loss with respect to this Tensor. (dLoss/dThisTensor).
        Used by optimizers to update self.data.

    """
    def __init__(self, data: Union[np.ndarray, list, int, float], requires_grad: bool = False):
        self.requires_grad = requires_grad
        if isinstance(data, Tensor): #copy attributes from the other tensor
            self.data = data.data.astype(np.float32, copy=True)
            self.requires_grad = data.requires_grad
        else:
            self.data = np.array(data, dtype=np.float32)

    #Basic arithmetic operators

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

    def __matmul__(self, other): # t1 @ t2 matrix multiplication
        # for 0-D, just multiples the values, same as __mul__
        # for 1-D, dot product of both arrays
        # for 2-D, must match shapes (n, k) @ (k, m) -> (n, m)
        # for N-D, tensors of shapes (..., n, k) @ (..., k, m) -> (..., n, m) as long as (...) matches or broadcasts
        if isinstance(other, Tensor):
            return Tensor(self.data @ other.data)
        return Tensor(self.data @ other)

    def __pow__(self, power): # t ** n
        #TODO: Right now, we lose requires_grad, but in the future (when we implement grad) this needs to be updated
        return Tensor(self.data ** power)

    def __neg__(self): # -t
        return Tensor(-1 * self.data)

    def __radd__(self, other): # other + t
        if isinstance(other, Tensor):
            return Tensor(other.data + self.data)
        return Tensor(other + self.data)

    def __rsub__(self, other): # other - t
        if isinstance(other, Tensor):
            return Tensor(other.data - self.data)
        return Tensor(other - self.data)

    def __rmul__(self, other): # other * t
        if isinstance(other, Tensor):
            return Tensor(other.data * self.data)
        return Tensor(other * self.data)

    def __rtruediv__(self, other): # other / t
        if isinstance(other, Tensor):
            return Tensor(other.data / self.data)
        return Tensor(other / self.data)

    #Properties

    @property
    def shape(self):
        return self.data.shape

    @property
    def size(self):
        return self.data.size

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def dtype(self):
        return self.data.dtype

    #String/Utility operators

    def __repr__(self): # print(t)
        s = np.array2string(
            self.data,
            separator=', ',
            threshold=1000,   # when to start using "..."
        )
        return f"Tensor(\n{s},\n\n dtype={self.data.dtype}, shape={self.data.shape}, requires_grad{self.requires_grad})"

    def __str__(self): #str(t)
        return self.__repr__(self)

    def copy(self):
        return Tensor(self)

    def to_numpy(self):
        return np.array(self.data)


    #Comparation operators

    def __eq__(self, other):
        if isinstance(other, Tensor):
            other_data = other.data
        else:
            other_data = other
        return self.data == other_data #returns element wise comparison

    def __ne__(self, other):
        if isinstance(other, Tensor):
            other_data = other.data
        else:
            other_data = other
        return self.data != other_data

    def __lt__(self, other):
        if isinstance(other, Tensor):
            other_data = other.data
        else:
            other_data = other
        return self.data < other_data # returns boolean array with element-wise comparison
    
    def __le__(self, other):
        if isinstance(other, Tensor):
            other_data = other.data
        else:
            other_data = other
        return self.data <= other_data

    def __gt__(self, other):
        if isinstance(other, Tensor):
            other_data = other.data
        else:
            other_data = other
        return self.data > other_data

    def __ge__(self, other):
        if isinstance(other, Tensor):
            other_data = other.data
        else:
            other_data = other
        return self.data >= other_data

    #Aggregate methods
    def sum(self, axis = None, keepdims = False, dtype = None): # t1.sum()
        result_data = self.data.sum(axis = axis, keepdims = keepdims, dtype = dtype)
        return Tensor(result_data, requires_grad=self.requires_grad)

    def item(self):
        #This is not an aggregate method, but it is used in combination with aggregate methods
        #to get the scalar value.
        if self.shape != ():
            raise ValueError("Can only call .item() on scalar tensors")
        return self.data.item()

    def mean(self, axis = None, keepdims = False):
        result_data = self.data.mean(axis = axis, keepdims = keepdims)
        return Tensor(result_data, requires_grad = self.requires_grad)

    def std(self, axis = None, keepdims = False): # Sum(0 <= i < n)[( (self.mean(axis = axis) - Xi)**2 ) / n]
        result_data = self.data.std(axis=axis, keepdims=keepdims)
        return Tensor(result_data, requires_grad=self.requires_grad)

    def var(self, axis = None, keepdims = False): # t1.std() ** 2
        result_data = self.data.var(axis=axis, keepdims=keepdims)
        return Tensor(result_data, requires_grad=self.requires_grad)

    def min(self, axis = None, keepdims = False):
        result_data = self.data.min(axis=axis, keepdims=keepdims)
        return Tensor(result_data, requires_grad=self.requires_grad)

    def max(self, axis = None, keepdims = False):
        result_data = self.data.max(axis=axis, keepdims=keepdims)
        return Tensor(result_data, requires_grad=self.requires_grad)