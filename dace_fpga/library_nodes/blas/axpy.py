# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.

import dace.library
from dace.transformation.transformation import ExpandTransformation
from dace import SDFG, SDFGState
from dace.libraries.blas.nodes.axpy import Axpy, ExpandAxpyVectorized


@dace.library.register_expansion(Axpy, "fpga")
class ExpandAxpyFpga(ExpandTransformation):
    """
    FPGA expansion which uses the generic implementation, but sets the map
    schedule to be executed on FPGA.
    """

    environments = []

    @staticmethod
    def expansion(node, parent_state: SDFGState, parent_sdfg: SDFG, **kwargs):
        """
        :param node: Node to expand.
        :param parent_state: State that the node is in.
        :param parent_sdfg: SDFG that the node is in.
        """
        return ExpandAxpyVectorized.expansion(node,
                                              parent_state,
                                              parent_sdfg,
                                              schedule=dace.ScheduleType.FPGA_Device,
                                              **kwargs)
