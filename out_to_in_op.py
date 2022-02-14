from aesara.graph import Op, Apply
from aesara import tensor as at
import numpy as np

from data import noteStateSingleToInputForm


class OutputFormToInputFormOp(Op):
    # Properties attribute
    __props__ = ()

    def make_node(self, state, time):
        state = at.as_tensor_variable(state)
        time = at.as_tensor_variable(time)
        return Apply(self, [state, time], [at.bmatrix()])

    # Python implementation:
    def perform(self, node, inputs_storage, output_storage):
        state, time = inputs_storage
        output_storage[0][0] = np.array(
            noteStateSingleToInputForm(state, time), dtype="int8"
        )
