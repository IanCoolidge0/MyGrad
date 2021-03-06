import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given
from hypothesis.extra import numpy as hnp

from mygrad import Tensor
from mygrad.errors import InvalidGradient
from mygrad.operation_base import Operation
from tests.utils import does_not_raise


class OldOperation(Operation):
    """Implements old version of MyGrad back-propagation"""

    def __call__(self, a):
        self.variables = (a,)
        return a.data

    def backward_var(self, grad, index, **kwargs):
        self.variables[index].backward(grad)


def old_op(a):
    return Tensor._op(OldOperation, a)


@given(constant=st.booleans(), arr=hnp.arrays(dtype=float, shape=hnp.array_shapes()),
       op_before=st.booleans(), op_after=st.booleans())
def test_backpropping_non_numeric_gradient_raises(constant: bool, arr: np.ndarray, op_before: bool,
                                                  op_after):
    x = Tensor(arr, constant=constant)

    if op_before:
        x += 1

    x = old_op(x)

    if op_after:
        x = x * 2

    # if constant tensor, backprop should not be triggered - no exception raised
    with (pytest.raises(InvalidGradient) if not constant else does_not_raise()):
        x.backward()
