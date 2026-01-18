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

class TestTensorCopy:
    def test_tensor_copying(self):
        t1 = Tensor([1, 2, 3], requires_grad=True)
        t2 = t1.copy()
        
        assert t1 is not t2 #they are different objects
        assert t1.data is not t2.data #with different data objects
    
        assert np.array_equal(t1.data, t2.data) #but have the same values
        assert t1.requires_grad == t2.requires_grad
        
        #mutating t2.data does not affect t1.data
        t2.data[0] = 999
        assert t1.data[0] == 1
        assert t2.data[0] == 999

class TestTensorToNumpy():
    def test_to_numpy(self):
        t1 = Tensor([1.5, 2.5, 3.5])
        arr = t1.to_numpy()

        assert isinstance(arr, np.ndarray)
        assert np.array_equal(arr, np.array([1.5, 2.5, 3.5], dtype=np.float32))
        assert arr.shape == t1.shape
        assert arr.dtype == np.float32


    def test_to_numpy_scalar(self):
        t = Tensor(5.0)
        arr = t.to_numpy()

        assert isinstance(arr, np.ndarray)
        assert arr.shape == ()
        assert arr.item() == 5.0


    def test_to_numpy_multidimensional(self):
        t = Tensor([[1, 2], [3, 4]])
        arr = t.to_numpy()

        assert isinstance(arr, np.ndarray)
        assert arr.shape == (2, 2)
        assert np.array_equal(arr, np.array([[1, 2], [3, 4]], dtype=np.float32))

class TestTensorEqual():
    def test_eq_tensor_vs_tensor(self):
        t1 = Tensor([1, 2, 3])
        t2 = Tensor([1, 2, 3])
        result = t1 == t2
        assert np.array_equal(result, np.array([True, True, True]))

    def test_eq_tensor_vs_tensor_not_equal(self):
        t1 = Tensor([1, 2, 3])
        t2 = Tensor([1, 2, 4])
        result = t1 == t2
        assert np.array_equal(result, np.array([True, True, False]))

    def test_eq_tensor_vs_scalar(self):
        t = Tensor([1, 2, 3])
        result = t == 2
        assert np.array_equal(result, np.array([False, True, False]))

    def test_ne_tensor_vs_tensor(self):
        t1 = Tensor([1, 2, 3])
        t2 = Tensor([1, 2, 3])
        result = t1 != t2
        assert np.array_equal(result, np.array([False, False, False]))

    def test_ne_tensor_vs_tensor_not_equal(self):
        t1 = Tensor([1, 2, 3])
        t2 = Tensor([1, 2, 4])
        result = t1 != t2
        assert np.array_equal(result, np.array([False, False, True]))

    def test_ne_tensor_vs_scalar(self):
        t = Tensor([1, 2, 3])
        result = t != 2
        assert np.array_equal(result, np.array([True, False, True]))


class TestTensorLessThan():
    def test_lt_tensor_vs_tensor(self):
        t1 = Tensor([1, 2, 3])
        t2 = Tensor([2, 2, 2])
        result = t1 < t2
        assert np.array_equal(result, np.array([True, False, False]))

    def test_lt_tensor_vs_scalar(self):
        t = Tensor([1, 2, 3])
        result = t < 2
        assert np.array_equal(result, np.array([True, False, False]))

    def test_lt_scalar(self):
        t = Tensor(1.5)
        result = t < 2.0
        assert result.item() == True

class TestTensorLessThanOrEqualTo():
    def test_le_tensor_vs_tensor(self):
        t1 = Tensor([1, 2, 3])
        t2 = Tensor([2, 2, 2])
        result = t1 <= t2
        assert np.array_equal(result, np.array([True, True, False]))

    def test_le_tensor_vs_scalar(self):
        t = Tensor([1, 2, 3])
        result = t <= 2
        assert np.array_equal(result, np.array([True, True, False]))

    def test_le_scalar(self):
        t = Tensor(2.0)
        result = t <= 2.0
        assert result.item() == True

class TestTensorGreaterThan():
    def test_gt_tensor_vs_tensor(self):
        t1 = Tensor([1, 2, 3])
        t2 = Tensor([2, 2, 2])
        result = t1 > t2
        assert np.array_equal(result, np.array([False, False, True]))

    def test_gt_tensor_vs_scalar(self):
        t = Tensor([1, 2, 3])
        result = t > 2
        assert np.array_equal(result, np.array([False, False, True]))

    def test_gt_scalar(self):
        t = Tensor(3.0)
        result = t > 2.0
        assert result.item() == True

class TestTensorGreaterThanOrEqualTo():
    def test_ge_tensor_vs_tensor(self):
        t1 = Tensor([1, 2, 3])
        t2 = Tensor([2, 2, 2])
        result = t1 >= t2
        assert np.array_equal(result, np.array([False, True, True]))

    def test_ge_tensor_vs_scalar(self):
        t = Tensor([1, 2, 3])
        result = t >= 2
        assert np.array_equal(result, np.array([False, True, True]))

    def test_ge_scalar(self):
        t = Tensor(2.0)
        result = t >= 2.0
        assert result.item() == True

class TestTensorSumMethod():
    def test_sum_all(self):
        t = Tensor([1, 2, 3])
        result = t.sum()
        assert np.array_equal(result.data, np.array(6, dtype=np.float32))

    def test_sum_axis_0(self):
        t = Tensor([[1, 2], [3, 4]])
        result = t.sum(axis=0)
        assert np.array_equal(result.data, np.array([4, 6], dtype=np.float32))

    def test_sum_keepdims(self):
        t = Tensor([[1, 2], [3, 4]])
        result = t.sum(axis=1, keepdims=True)
        assert result.shape == (2, 1)
        assert np.array_equal(result.data, np.array([[3], [7]], dtype=np.float32))

    def test_sum_scalar(self):
        t = Tensor(5.0)
        result = t.sum()
        assert result.data.item() == 5.0

    def test_sum_requires_grad(self):
        t = Tensor([1, 2], requires_grad=True)
        result = t.sum()
        assert result.requires_grad == True

class TestTensorMeanMethod():
    def test_mean_all(self):
        t = Tensor([1, 2, 3])
        result = t.mean()
        assert np.array_equal(result.data, np.array(2.0, dtype=np.float32))

    def test_mean_axis_0(self):
        t = Tensor([[1, 2], [3, 4]])
        result = t.mean(axis=0)
        assert np.array_equal(result.data, np.array([2.0, 3.0], dtype=np.float32))

    def test_mean_axis_1(self):
        t = Tensor([[1, 2], [3, 4]])
        result = t.mean(axis=1)
        assert np.array_equal(result.data, np.array([1.5, 3.5], dtype=np.float32))

    def test_mean_keepdims(self):
        t = Tensor([[1, 2], [3, 4]])
        result = t.mean(axis=1, keepdims=True)
        assert result.shape == (2, 1)
        assert np.array_equal(result.data, np.array([[1.5], [3.5]], dtype=np.float32))

    def test_mean_scalar(self):
        t = Tensor(5.0)
        result = t.mean()
        assert result.data.item() == 5.0

    def test_mean_requires_grad(self):
        t = Tensor([1, 2], requires_grad=True)
        result = t.mean()
        assert result.requires_grad == True


class TestTensorStdMethod():
    def test_std_all(self):
        t = Tensor([1, 2, 3], requires_grad=True)
        result = t.std()
        expected = np.array([1, 2, 3], dtype=np.float32).std()
        assert np.isclose(result.data.item(), expected)
        assert result.requires_grad is True

    def test_std_axis_0(self):
        t = Tensor([[1, 2], [3, 4]])
        result = t.std(axis=0)
        expected = np.array([[1, 2], [3, 4]], dtype=np.float32).std(axis=0)
        assert np.allclose(result.data, expected)

    def test_std_keepdims(self):
        t = Tensor([[1, 2], [3, 4]])
        result = t.std(axis=1, keepdims=True)
        expected = np.array([[1, 2], [3, 4]], dtype=np.float32).std(axis=1, keepdims=True)
        assert result.shape == (2, 1)
        assert np.allclose(result.data, expected)

class TestTensorVarMethod():
    def test_var_all(self):
        t = Tensor([1, 2, 3], requires_grad=True)
        result = t.var()
        expected = np.array([1, 2, 3], dtype=np.float32).var()
        assert np.isclose(result.data.item(), expected)
        assert result.requires_grad is True

    def test_var_axis_0(self):
        t = Tensor([[1, 2], [3, 4]])
        result = t.var(axis=0)
        expected = np.array([[1, 2], [3, 4]], dtype=np.float32).var(axis=0)
        assert np.allclose(result.data, expected)

    def test_var_keepdims(self):
        t = Tensor([[1, 2], [3, 4]])
        result = t.var(axis=1, keepdims=True)
        expected = np.array([[1, 2], [3, 4]], dtype=np.float32).var(axis=1, keepdims=True)
        assert result.shape == (2, 1)
        assert np.allclose(result.data, expected)

class TestTensorMinMethod():
    def test_min_all(self):
        t = Tensor([3, 1, 2])
        result = t.min()
        assert result.data.item() == 1.0

    def test_min_axis_0(self):
        t = Tensor([[3, 1], [2, 4]])
        result = t.min(axis=0)
        assert np.array_equal(result.data, np.array([2.0, 1.0]))

    def test_min_keepdims(self):
        t = Tensor([[3, 1], [2, 4]])
        result = t.min(axis=1, keepdims=True)
        assert result.shape == (2, 1)
        assert np.array_equal(result.data, np.array([[1.0], [2.0]]))

class TestTensorMaxMethod():
    def test_max_all(self):
        t = Tensor([3, 1, 2])
        result = t.max()
        assert result.data.item() == 3.0

    def test_max_axis_0(self):
        t = Tensor([[3, 1], [2, 4]])
        result = t.max(axis=0)
        assert np.array_equal(result.data, np.array([3.0, 4.0]))

    def test_max_keepdims(self):
        t = Tensor([[3, 1], [2, 4]])
        result = t.max(axis=1, keepdims=True)
        assert result.shape == (2, 1)
        assert np.array_equal(result.data, np.array([[3.0], [4.0]]))

class TestTensorReshapeMethod():
    def test_reshape_1d_to_2d(self):
        t = Tensor([1, 2, 3, 4, 5, 6])
        result = t.reshape(2, 3)
        assert result.shape == (2, 3)
        assert np.array_equal(result.data.flatten(), np.array([1, 2, 3, 4, 5, 6], dtype=np.float32))

    def test_reshape_infer_dimension(self):
        t = Tensor([1, 2, 3, 4])
        result = t.reshape(2, -1)
        assert result.shape == (2, 2)

    def test_reshape_scalar(self):
        t = Tensor(5.0)
        result = t.reshape(1)
        assert result.shape == (1,)
        assert result.data.item() == 5.0



class TestTensorFlattenMethod():
    def test_flatten_2d_to_1d(self):
        t = Tensor([[1, 2], [3, 4]])
        result = t.flatten()
        assert result.shape == (4,)
        assert np.array_equal(result.data, np.array([1, 2, 3, 4], dtype=np.float32))

    def test_flatten_scalar(self):
        t = Tensor(5.0)
        result = t.flatten()
        assert result.shape == (1, )
        assert result.data.item() == 5.0

    def test_flatten_already_1d(self):
        t = Tensor([1, 2, 3])
        result = t.flatten()
        assert np.array_equal(result.data, t.data)

    def test_reshape_requires_grad(self):
        t = Tensor([1, 2, 3], requires_grad=True)
        result = t.reshape(1, 3)
        assert result.requires_grad is True
