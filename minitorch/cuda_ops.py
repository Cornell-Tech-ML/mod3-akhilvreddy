# type: ignore
# Currently pyright doesn't support numba.cuda

from typing import Callable, Optional, TypeVar, Any

import numba
from numba import cuda
from numba.cuda import jit as _jit
from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Shape,
    Storage,
    Strides,
    TensorData,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

FakeCUDAKernel = Any

# This code will CUDA compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.

Fn = TypeVar("Fn")


def device_jit(fn: Fn, **kwargs) -> Fn:
    return _jit(device=True, **kwargs)(fn)  # type: ignore


def jit(fn, **kwargs) -> FakeCUDAKernel:
    return _jit(**kwargs)(fn)  # type: ignore


to_index = device_jit(to_index)
index_to_position = device_jit(index_to_position)
broadcast_index = device_jit(broadcast_index)

THREADS_PER_BLOCK = 32


class CudaOps(TensorOps):
    cuda = True

    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """See `tensor_ops.py`"""
        cufn: Callable[[float], float] = device_jit(fn)
        f = tensor_map(cufn)

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)

            # Instantiate and run the cuda kernel.
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
            f[blockspergrid, threadsperblock](*out.tuple(), out.size, *a.tuple())  # type: ignore
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_zip(cufn)

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + (threadsperblock - 1)) // threadsperblock
            f[blockspergrid, threadsperblock](  # type: ignore
                *out.tuple(), out.size, *a.tuple(), *b.tuple()
            )
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_reduce(cufn)

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = (a.shape[dim] - 1) // 1024 + 1
            out_a = a.zeros(tuple(out_shape))

            threadsperblock = 1024
            blockspergrid = out_a.size
            f[blockspergrid, threadsperblock](  # type: ignore
                *out_a.tuple(), out_a.size, *a.tuple(), dim, start
            )

            return out_a

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        # Make these always be a 3 dimensional multiply
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        # One block per batch, extra rows, extra col
        blockspergrid = (
            (out.shape[1] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            (out.shape[2] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            out.shape[0],
        )
        threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)

        tensor_matrix_multiply[blockspergrid, threadsperblock](
            *out.tuple(), out.size, *a.tuple(), *b.tuple()
        )

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implement


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """CUDA higher-order tensor map function. ::

      fn_map = tensor_map(fn)
      fn_map(out, ... )

    Args:
    ----
        fn: function mappings floats-to-floats to apply.

    Returns:
    -------
        Tensor map function.

    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        in_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

        if i < out_size:  # Boundary check to ensure we're within tensor bounds
            # Convert `i` to `out_index` in `out_shape`
            to_index(i, out_shape, out_index)

            # Broadcast `out_index` to `in_index` to get corresponding input position
            broadcast_index(out_index, out_shape, in_shape, in_index)

            # Convert multi-dimensional indices to flat positions
            out_pos = index_to_position(out_index, out_strides)
            in_pos = index_to_position(in_index, in_strides)

            # Apply the function to the input element and store it in the output
            out[out_pos] = fn(in_storage[in_pos])

    return cuda.jit()(_map)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """CUDA higher-order tensor zipWith (or map2) function ::

      fn_zip = tensor_zip(fn)
      fn_zip(out, ...)

    Args:
    ----
        fn: function mappings two floats to float to apply.

    Returns:
    -------
        Tensor zip function.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        a_index = cuda.local.array(MAX_DIMS, numba.int32)
        b_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

        # Check boundary condition to ensure we're within bounds
        if i < out_size:
            # Convert flat index `i` to multi-dimensional `out_index` in `out_shape`
            to_index(i, out_shape, out_index)

            # Broadcast `out_index` to `a_index` and `b_index` based on shapes
            broadcast_index(out_index, out_shape, a_shape, a_index)
            broadcast_index(out_index, out_shape, b_shape, b_index)

            # Convert multi-dimensional indices to flat positions
            out_pos = index_to_position(out_index, out_strides)
            a_pos = index_to_position(a_index, a_strides)
            b_pos = index_to_position(b_index, b_strides)

            # Apply the function to the corresponding elements in a and b and store in out
            out[out_pos] = fn(a_storage[a_pos], b_storage[b_pos])

    return cuda.jit()(_zip)  # type: ignore


def _sum_practice(out: Storage, a: Storage, size: int) -> None:
    """This is a practice sum kernel to prepare for reduce.

    Given an array of length $n$ and out of size $n // \text{blockDIM}$
    it should sum up each blockDim values into an out cell.

    $[a_1, a_2, ..., a_{100}]$

    |

    $[a_1 +...+ a_{31}, a_{32} + ... + a_{64}, ... ,]$

    Note: Each block must do the sum using shared memory!

    Args:
    ----
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        size (int):  length of a.

    """
    BLOCK_DIM = 32

    # Allocate shared memory for the block
    cache = cuda.shared.array(BLOCK_DIM, numba.float64)

    # Compute the global index and the thread's position within the block
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    pos = cuda.threadIdx.x

    # Each thread loads one element from `a` into shared memory if within bounds
    if i < size:
        cache[pos] = a[i]
    else:
        cache[pos] = 0.0  # Fill out-of-bounds positions with 0
    cuda.syncthreads()

    # Perform parallel reduction within the block
    stride = 1
    while stride < BLOCK_DIM:
        if pos % (2 * stride) == 0 and pos + stride < BLOCK_DIM:
            cache[pos] += cache[pos + stride]
        cuda.syncthreads()
        stride *= 2

    # The first thread in each block writes the result to the output
    if pos == 0:
        out[cuda.blockIdx.x] = cache[0]


jit_sum_practice = cuda.jit()(_sum_practice)


def sum_practice(a: Tensor) -> TensorData:
    (size,) = a.shape
    threadsperblock = THREADS_PER_BLOCK
    blockspergrid = (size // THREADS_PER_BLOCK) + 1
    out = TensorData([0.0 for i in range(2)], (2,))
    out.to_cuda_()
    jit_sum_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, size
    )
    return out


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """CUDA higher-order tensor reduce function.

    Args:
    ----
        fn: reduction function maps two floats to float.

    Returns:
    -------
        Tensor reduce function.

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
        reduce_value: float,
    ) -> None:
        BLOCK_DIM = 1024
        cache = cuda.shared.array(BLOCK_DIM, numba.float64)
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        out_pos = cuda.blockIdx.x
        pos = cuda.threadIdx.x

        # Set initial reduction value in shared memory
        cache[pos] = reduce_value

        # For each element in `out`, reduce along `reduce_dim` using `fn`
        to_index(out_pos, out_shape, out_index)
        for i in range(pos, a_shape[reduce_dim], BLOCK_DIM):
            out_index[reduce_dim] = i
            a_pos = index_to_position(out_index, a_strides)
            cache[pos] = fn(cache[pos], a_storage[a_pos])

        # Synchronize threads within the block
        cuda.syncthreads()

        # Perform parallel reduction in shared memory
        stride = BLOCK_DIM // 2
        while stride > 0:
            if pos < stride:
                cache[pos] = fn(cache[pos], cache[pos + stride])
            cuda.syncthreads()
            stride //= 2

        # The first thread in each block writes the reduction result to `out`
        if pos == 0:
            out[out_pos] = cache[0]

    return jit(_reduce)  # type: ignore


def _mm_practice(out: Storage, a: Storage, b: Storage, size: int) -> None:
    """This is a practice square MM kernel to prepare for matmul.

    Given a storage `out` and two storage `a` and `b`. Where we know
    both are shape [size, size] with strides [size, 1].

    Size is always < 32.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Compute

    ```
     for i:
         for j:
              for k:
                  out[i, j] += a[i, k] * b[k, j]
    ```

    Args:
    ----
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        b (Storage): storage for `b` tensor.
        size (int): size of the square

    """
    BLOCK_DIM = 32

    # Allocate shared memory for `a` and `b` blocks
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    # Calculate row and column indexes in the output matrix
    row = cuda.threadIdx.y
    col = cuda.threadIdx.x
    block_row = cuda.blockIdx.y * BLOCK_DIM
    block_col = cuda.blockIdx.x * BLOCK_DIM

    # Initialize the output value
    temp_sum = 0.0

    # Loop over each sub-matrix in `a` and `b` for block multiplication
    for m in range((size + BLOCK_DIM - 1) // BLOCK_DIM):
        # Load data from global memory to shared memory
        if block_row + row < size and m * BLOCK_DIM + col < size:
            a_shared[row, col] = a[(block_row + row) * size + (m * BLOCK_DIM + col)]
        else:
            a_shared[row, col] = 0.0

        if m * BLOCK_DIM + row < size and block_col + col < size:
            b_shared[row, col] = b[(m * BLOCK_DIM + row) * size + (block_col + col)]
        else:
            b_shared[row, col] = 0.0

        # Synchronize to make sure the data is loaded
        cuda.syncthreads()

        # Perform the multiplication and accumulate the result
        for k in range(BLOCK_DIM):
            temp_sum += a_shared[row, k] * b_shared[k, col]

        # Synchronize to ensure all threads have completed the computation
        cuda.syncthreads()

    # Write the result to global memory
    if block_row + row < size and block_col + col < size:
        out[(block_row + row) * size + (block_col + col)] = temp_sum


jit_mm_practice = cuda.jit(_mm_practice)


def mm_practice(a: Tensor, b: Tensor) -> TensorData:
    (size, _) = a.shape
    threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK)
    blockspergrid = 1
    out = TensorData([0.0 for i in range(size * size)], (size, size))
    out.to_cuda_()
    jit_mm_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, b._tensor._storage, size
    )
    return out


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """CUDA tensor matrix multiply function.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Should work for any tensor shapes that broadcast as long as ::

    ```python
    assert a_shape[-1] == b_shape[-2]
    ```
    Returns:
        None : Fills in `out`
    """
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0
    # Batch dimension - fixed
    batch = cuda.blockIdx.z

    BLOCK_DIM = 32
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    # The final position c[i, j]
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    # The local position in the block.
    pi = cuda.threadIdx.x
    pj = cuda.threadIdx.y

    # Initialize the accumulation register
    c_sum = 0.0

    # Assuming the last two dimensions are the matrix dimensions
    M = a_shape[-2]
    K = a_shape[-1]
    N = b_shape[-1]

    # Calculate the number of tiles
    num_tiles = math.ceil(K / BLOCK_DIM)

    # Calculate batch offsets with broadcasting
    a_batch_offset = batch * a_batch_stride
    b_batch_offset = batch * b_batch_stride

    for tile in range(num_tiles):
        # Calculate the global k index for this tile
        k = tile * BLOCK_DIM + pi

        # Load a_shared
        if (i < M) and (k < K):
            a_index = a_batch_offset
            # Calculate the multi-dimensional index for a
            a_index += (i * a_strides[-2] + k * a_strides[-1])
            a_shared[pi, pj] = a_storage[a_index]
        else:
            a_shared[pi, pj] = 0.0  # Padding with zero for out-of-bounds

        # Load b_shared
        k_b = tile * BLOCK_DIM + pj
        if (k_b < K) and (j < N):
            b_index = b_batch_offset
            # Calculate the multi-dimensional index for b
            b_index += (k_b * b_strides[-2] + j * b_strides[-1])
            b_shared[pi, pj] = b_storage[b_index]
        else:
            b_shared[pi, pj] = 0.0  # Padding with zero for out-of-bounds

        # Synchronize to make sure the tiles are loaded
        cuda.syncthreads()

        # Perform the multiplication for this tile
        for t in range(BLOCK_DIM):
            c_sum += a_shared[pi, t] * b_shared[t, pj]

        # Synchronize before loading the next tile
        cuda.syncthreads()

    # Write the result to global memory
    if (i < M) and (j < N):
        out_index = batch
        # Calculate the multi-dimensional index for out
        out_index += (i * out_strides[-2] + j * out_strides[-1])
        out[out_index] = c_sum


tensor_matrix_multiply = jit(_tensor_matrix_multiply)
