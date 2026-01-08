import pytest
import numpy as np
from math import isclose
from prisml import Tensor

FLOAT_ARRAY = np.array([1.5, -2.0, 3.14159, 0.0])

class TestTensorCreation:
    """Test Tensor Initialization."""

    def test_creation_from_scalar(self):
        t = Tensor(1)
        assert t.data.shape == ()
        assert t.data.dtype == np.float32
        assert t.data == 1

    def test_creation_from_int_list(self):
        t = Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 0])
        assert t.data.shape == (10, )
        assert t.data.dtype == np.float32
        assert np.array_equal(t.data, np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 0]))

    def test_creation_from_float_list(self):
        t = Tensor(FLOAT_ARRAY)
        assert t.data.shape == (4,)
        #use allclose to compare floats since array_equal fails with floating point precision issues
        assert np.allclose(t.data, FLOAT_ARRAY)
        assert t.data.dtype == np.float32

    def test_creation_from_tensor(self):
        t1 = Tensor(FLOAT_ARRAY)
        t2 = Tensor(t1)
        assert t2.data.shape == (4,)
        assert np.allclose(t2.data, FLOAT_ARRAY)
        assert t2.data.dtype == np.float32
        assert t2.requires_grad == t1.requires_grad



class TestTensorShape:
    """Test Tensor Shape Property."""

    def test_scalar_shape(self):
        t = Tensor(1)
        assert t.shape == ()

    def test_1d_array_shape(self):
        t = Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 0])
        assert t.shape == (10,)

    def test_2d_matrix_array_shape(self):
        t = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        assert t.shape == (3, 3)

    def test_3d_matrx_array_shape(self):
        t = Tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
        assert t.shape == (1, 3, 3)

class TestTensorAddition:
    """Test Tensor Addition"""

    def test_scalar_plus_tensor_addition(self):
        t = Tensor(1)
        result = t + 3
        assert result.shape == ()
        assert result.data == 4

    def test_scalar_tensors_addition(self):
        t1 = Tensor(1)
        t2 = Tensor(2)
        result = t1 + t2
        assert result.shape == ()
        assert result.data == 3

    def test_1d_array_tensor_addition(self):
        t1 = Tensor([1, 1, 1, 1, 1])
        t2 = Tensor([0, 1, 2, 3, 4])
        result = t1 + t2
        assert result.shape == (5, )
        assert np.array_equal(result.data, np.array([1, 2, 3, 4, 5]))

    def test_1d_float_array_tensor_addition(self):
        t1 = Tensor(FLOAT_ARRAY)
        t2 = Tensor(2 * FLOAT_ARRAY)
        result = t1 + t2
        assert result.shape == (4, )
        assert np.allclose(result.data, np.array(3 * FLOAT_ARRAY))

    def test_2d_array_tensor_addition(self):
        t1 = Tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        t2 = Tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
        result = t1 + t2
        assert result.shape == (3, 3)
        assert np.array_equal(result.data, np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))

    def test_3d_array_tensor_addition(self):
        t1 = Tensor([[[1, 1, 1], [1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1], [1, 1, 1]]])
        t2 = Tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8]], [[0, 1, 2], [3, 4, 5], [6, 7, 8]], [[0, 1, 2], [3, 4, 5], [6, 7, 8]]])  
        result = t1 + t2
        assert result.shape == (3, 3, 3)
        assert np.array_equal(result.data, np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[1, 2, 3], [4, 5, 6], [7, 8, 9]]]))

    def test_scalar_plus_array(self):
        t1 = Tensor(1)
        t2 = Tensor([1, 2, 4])
        result = t1 + t2
        assert result.shape == (3,)
        assert np.array_equal(result.data, np.array([2, 3, 5]))

    def test_addition_broadcast_compatible_shapes(self):
        t1 = Tensor([1, 2])
        t2 = Tensor([[10, 20], [30, 40]])
        result = t1 + t2
        assert result.shape == (2, 2)
        assert np.array_equal(result.data, np.array([[11, 22], [31, 42]]))

    def test_addition_broadcast_incompatible_shapes(self):
        t1 = Tensor([1, 2, 3])
        t2 = Tensor([[10, 20], [30, 40]])
        with pytest.raises(ValueError):
            result = t1 + t2

class TestTensorSubtraction:
    """Test Tensor Subtraction"""

    def test_scalar_minus_tensor_subtraction(self):
        t = Tensor(10)
        result = t - 2
        assert result.shape == ()
        assert result.data == 8

    def test_scalar_tensor_subtraction(self):
        t1 = Tensor(10)
        t2 = Tensor(2)
        result = t1 - t2
        assert result.shape == ()
        assert result.data == 8

    def test_1d_array_tensor_subtraction(self):
        t1 = Tensor([10, 10, 10, 10, 10])
        t2 = Tensor([0, 1, 2, 3, 4])
        result = t1 - t2
        assert result.shape == (5, )
        assert np.array_equal(result.data, np.array([10, 9, 8, 7, 6]))

    def test_1d_float_array_tensor_subtraction(self):
        t1 = Tensor(3 * FLOAT_ARRAY)
        t2 = Tensor(FLOAT_ARRAY)
        result = t1 - t2
        assert result.shape == (4, )
        assert np.allclose(result.data, np.array(2 * FLOAT_ARRAY))

    def test_2d_array_tensor_subtraction(self):
        t1 = Tensor([[10, 10, 10], [10, 10, 10], [10, 10, 10]])
        t2 = Tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
        result = t1 - t2
        assert result.shape == (3, 3)
        assert np.array_equal(result.data, np.array([[10, 9, 8], [7, 6, 5], [4, 3, 2]]))

    def test_3d_array_tensor_subtraction(self):
        t1 = Tensor([[[1, 1, 1], [1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1], [1, 1, 1]]])
        t2 = Tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8]], [[0, 1, 2], [3, 4, 5], [6, 7, 8]], [[0, 1, 2], [3, 4, 5], [6, 7, 8]]])  
        result = t1 - t2
        assert result.shape == (3, 3, 3)
        assert np.array_equal(result.data, np.array([[[1, 0, -1], [-2, -3, -4], [-5, -6, -7]], [[1, 0, -1], [-2, -3, -4], [-5, -6, -7]], [[1, 0, -1], [-2, -3, -4], [-5, -6, -7]]]))

    def test_scalar_minus_array(self):
        t1 = Tensor([1, 2, 4])
        t2 = Tensor(1)
        result = t1 - t2
        assert result.shape == (3,)
        assert np.array_equal(result.data, np.array([0, 1, 3]))

    def test_subtraction_broadcast_compatible_shapes(self):
        t1 = Tensor([[10, 20], [30, 40]])
        t2 = Tensor([1, 2])
        result = t1 - t2
        assert result.shape == (2, 2)
        assert np.array_equal(result.data, np.array([[9, 18], [29, 38]]))

    def test_subtraction_broadcast_incompatible_shapes(self):
        t1 = Tensor([1, 2, 3])
        t2 = Tensor([[10, 20], [30, 40]])
        with pytest.raises(ValueError):
            result = t1 - t2

class TestTensorMultiplication:
    """Test Tensor Multiplication"""

    def test_scalar_times_array_multiplication(self):
        t = Tensor([[1, 2, 3], [4, 5, 6]])
        result = 4 * t
        assert result.shape == (2, 3)
        assert np.array_equal(result.data, np.array([[4, 8, 12], [16, 20, 24]]))

    def test_scalar_times_array_multiplication(self):
        t1 = Tensor(4)
        t2 = Tensor([[1, 2, 3], [4, 5, 6]])
        result = t1 * t2
        assert result.shape == (2, 3)
        assert np.array_equal(result.data, np.array([[4, 8, 12], [16, 20, 24]]))

    def test_same_shape_array_multiplication(self):
        t1 = Tensor([1, 2, 3, 4, 5])
        t2 = Tensor([1, 2, 3, 4, 5])
        result = t1 * t2
        assert result.shape == (5, )
        assert np.array_equal(result.data, np.array([1, 4, 9, 16, 25]))

    def test_broadcastable_shape_array_multiplication(self):
        t1 = Tensor([1, 2, 3])
        t2 = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        result = t1 * t2
        assert result.shape == (3,3)
        assert np.array_equal(result.data, np.array([[1, 4, 9] , [4, 10, 18], [7, 16, 27]]))

    def test_non_broadcastable_shape_array_multiplication(self):
        t1 = Tensor([1, 2])
        t2 = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        with pytest.raises(ValueError):
            result = t1 * t2

class TestTensorDivision:
    """Test Tensor Division"""

    def test_scalar_divides_array(self):
        t = Tensor([[2, 2, 3], [4, 5, 6]])
        result = t / 2
        assert result.shape == (2, 3)
        assert np.allclose(result.data, np.array([[1, 1, 3/2], [2, 5/2, 3]]))

    def test_scalar_tensor_divides_array(self):
        t1 = Tensor(2)
        t2 = Tensor([[2, 2, 3], [4, 5, 6]])
        result = t2 / t1
        assert result.shape == (2, 3)
        assert np.allclose(result.data, np.array([[1, 1, 3/2], [2, 5/2, 3]]))

    def test_same_shape_array_division(self):
        t1 = Tensor([1, 2, 3, 4, 5])
        t2 = Tensor([3, 4, 5, 6, 7])
        result = t1 / t2
        assert result.shape == (5, )
        assert np.allclose(result.data, np.array([1/3, 0.5, 3/5, 2/3, 5/7]))

    def test_broadcastable_shape_array_division(self):
        t1 = Tensor([1, 2, 3])
        t2 = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        result = t2 / t1
        assert result.shape == (3,3)
        assert np.allclose(result.data, np.array([[1, 1, 1] , [4, 5/2, 2], [7, 4, 3]]))

    def test_non_broadcastable_shape_array_division(self):
        t1 = Tensor([1, 2])
        t2 = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        with pytest.raises(ValueError):
            result = t1 / t2
    
class TestTensorMatrixMultiplication:
    """Test Tensor Matrix Multiplication"""

    def test_matmul_vector_dot(self):
        # (3,) @ (3,) -> () because it does the dot product between both arrays and produces a scalar value
        t1 = Tensor([1, 2, 3])
        t2 = Tensor([4, 5, 6])
        
        result = t1 @ t2
        
        assert result.shape == () 
        assert result.data == 32

    def test_matmul_standard(self):
        # (2, 3) @ (3, 2) -> (2, 2)
        t1 = Tensor([[1, 2, 3], [4, 5, 6]])
        t2 = Tensor([[7, 8], [9, 10], [11, 12]])
        
        result = t1 @ t2
        
        assert result.shape == (2, 2)
        assert np.allclose(result.data, np.array([[58, 64], [139, 154]]))

    def test_matmul_matrix_vector(self):
        # (2, 3) @ (3,) -> (2,)
        t1 = Tensor([[1, 2, 3], [4, 5, 6]])
        t2 = Tensor([1, 0, 1])
        
        result = t1 @ t2
        
        assert result.shape == (2,)
        assert np.allclose(result.data, np.array([4, 10]))

    def test_matmul_batch_4d_with_broadcasting(self):
        # (1, 3, 4, 5) @ (2, 1, 5, 2) -> (2, 3, 4, 2)
        # The last two shape dims (4, 5) and (5, 2) already match the rule (..., n, k) @ (..., k, m).
        # Then, 1 is broadcast to match 2, for the first shape dim
        # And finally, 1 is broadcast to match 3, for the second shape dim
        
        t1 = Tensor(np.random.randn(1, 3, 4, 5))
        t2 = Tensor(np.random.randn(2, 1, 5, 2))
        
        result = t1 @ t2
        
        # Shape calculation:
        # (1, 3) broadcast with (2, 1) -> (2, 3)
        # (4, 5) @ (5, 2) -> (4, 2)
        assert result.shape == (2, 3, 4, 2)
        
        # Verify against NumPy's
        expected = np.matmul(t1.data, t2.data)
        assert np.allclose(result.data, expected)

    def test_matmul_batch_4d_single_matrix_broadcast(self):
        # (2, 3, 4, 5) @ (5, 2) -> (2, 3, 4, 2)
        # The (5, 2) matrix gets broadcast to (1, 1, 5, 2) then (2, 3, 5, 2)

        np.random.seed(42) 
        
        t1 = Tensor(np.random.randn(2, 3, 4, 5))
        t2 = Tensor(np.eye(5, 2))
        
        result = t1 @ t2
        
        assert result.shape == (2, 3, 4, 2)
        
        # Verify against NumPy
        expected = np.matmul(t1.data, t2.data)
        assert np.allclose(result.data, expected)

class TestTensorPowers():
    """Test Tensor Powers"""

    def test_pow_scalar_tensor(self):
        t = Tensor(4)

        result = t ** 2

        assert result.shape == ()
        assert result.data == 16

    def test_pow_scalar_float_tensor(self):
        t = Tensor(1 / 3)

        result = t ** 2

        assert result.shape == ()
        assert np.isclose(result.data.item(), np.float32(1 / 3) ** 2)

    def test_pow_vector_tensor(self):
        t = Tensor([1, 2, 3, 4])

        result = t ** 2

        assert result.shape == (4, )
        assert np.array_equal(result.data, np.array([1, 4, 9, 16]))

    def test_pow_3d_matrix_tensor(self):
        t = Tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])

        result = t ** 2

        assert result.shape == (1, 3, 3)
        assert np.array_equal(result.data, np.array([[[1, 4, 9], [16, 25, 36], [49, 64, 81]]]))

class TestTensorNeg():
    """Test Tensor Sign Inversion"""

    def test_neg_scalar_tensor(self):
        t = Tensor(2)

        result = -t

        assert result.shape == ()
        assert result.data == -2

    def test_pow_scalar_float_tensor(self):
        t = Tensor(1 / 3)

        result = -t

        assert result.shape == ()
        assert np.isclose(result.data.item(), -1 * np.float32(1 / 3))

    def test_pow_vector_tensor(self):
        t = Tensor([1, 2, 3, 4])

        result = -t

        assert result.shape == (4, )
        assert np.array_equal(result.data, np.array([-1, -2, -3, -4]))

    def test_pow_3d_matrix_tensor(self):
        t = Tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])

        result = -t

        assert result.shape == (1, 3, 3)
        assert np.array_equal(result.data, np.array([[[-1, -2, -3], [-4, -5, -6], [-7, -8, -9]]]))

class TestTensorRightAddition:
    """Test Tensor Right Addition"""

    def test_scalar_plus_tensor_right_addition(self):
        t = Tensor(2)
        result = 4 + t
        assert result.shape == ()
        assert result.data == 6

    def test_scalar_plus_array_tensor_right_addition(self):
        t = Tensor([1, 2, 3])
        result = 4 + t
        assert result.shape == (3, )
        assert np.array_equal(result.data, np.array([5, 6, 7]))

    def test_numpy_vector_plus_tensor_right_addition(self):
        t = Tensor(np.array([1, 2, 3]))
        result = 4 + t
        assert result.shape == (3, )
        assert np.array_equal(result.data, np.array([5, 6, 7]))

class TestTensorRightSubtraction:
    """Test Tensor Right Subtraction"""

    def test_scalar_plus_tensor_right_subtraction(self):
        t = Tensor(2)
        result = 4 - t
        assert result.shape == ()
        assert result.data == 2

    def test_scalar_plus_array_tensor_right_subtraction(self):
        t = Tensor([1, 2, 3])
        result = 4 - t
        assert result.shape == (3, )
        assert np.array_equal(result.data, np.array([3, 2, 1]))

    def test_numpy_vector_plus_tensor_right_subtraction(self):
        t = Tensor(np.array([1, 2, 3]))
        result = 4 - t
        assert result.shape == (3, )
        assert np.array_equal(result.data, np.array([3, 2, 1]))

class TestTensorRightMultiplication:
    """Test Tensor Right Multiplation"""

    def test_scalar_tensor_scalar_right_multiplication(self):
        t = Tensor(4)
        result = 2 * t
        assert result.shape == ()
        assert result.data == 8

    def test_scalar_tensor_array_right_multiplication(self):
        t = Tensor([1, 2, 3])
        result = 4 * t
        assert result.shape == (3,)
        assert np.array_equal(result.data, [4, 8, 12])

    def test_scalar_tensor_numpy_array_right_multiplication(self):
        t = Tensor(np.array([1, 2, 3]))
        result = -2 * t
        assert result.shape == (3, )
        assert np.array_equal(result.data, np.array([-2, -4, -6]))

class TestTensorRightTrueDivision:
    """Test Tensor Right True Division"""

    def test_scalar_tensor_scalar_right_division(self):
        t = Tensor(4)
        result = 8 / t
        assert result.shape == ()
        assert result.data == 2

    def test_scalar_tensor_array_right_division(self):
        t = Tensor([1, 2, 3])
        result = 4 / t
        assert result.shape == (3,)
        assert np.allclose(result.data, [4, 2, 4 / 3])

    def test_scalar_tensor_numpy_array_right_division(self):
        t = Tensor(np.array([1, 2, 3]))
        result = -2 / t
        assert result.shape == (3, )
        assert np.allclose(result.data, np.array([-2, -1, -2 / 3]))