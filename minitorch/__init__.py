"""
This is the top-level module for the `minitorch` package.

The `minitorch` package provides a minimalistic deep learning framework that includes:
- Tensor operations (`tensor.py`)
- Automatic differentiation (`autodiff.py`)
- Neural network modules (`module.py`)
- Optimizers (`optim.py`)
- Dataset utilities (`datasets.py`)
- GPU support via CUDA (`cuda_ops.py`)

This package is designed for educational purposes and to help users learn the
fundamentals of building a deep learning framework from scratch.

Imports from this module make core components directly accessible.
"""


from .testing import MathTest, MathTestVariable  # type: ignore # noqa: F401,F403
from .fast_ops import *  # noqa: F401,F403
from .cuda_ops import *  # noqa: F401,F403
from .tensor_data import *  # noqa: F401,F403
from .tensor_functions import *  # noqa: F401,F403
from .tensor_ops import *  # noqa: F401,F403
from .scalar import *  # noqa: F401,F403
from .scalar_functions import *  # noqa: F401,F403
from .module import *  # noqa: F401,F403
from .autodiff import *  # noqa: F401,F403
from .module import *  # noqa: F401,F403
from .module import *  # noqa: F401,F403
from .autodiff import *  # noqa: F401,F403
from .tensor import *  # noqa: F401,F403
from .datasets import *  # noqa: F401,F403
from .testing import *  # noqa: F401,F403
from .optim import *  # noqa: F401,F403
from . import fast_ops, cuda_ops  # noqa: F401,F403