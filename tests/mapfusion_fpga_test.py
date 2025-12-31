# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from dace_fpga.fpga_testing import fpga_test, xilinx_test
from dace.transformation.dataflow import MapFusionVertical
from dace_fpga.transformations import FPGATransformSDFG
import numpy as np
from dace.config import set_temporary
import dace


@dace.program
def multiple_fusions(A: dace.float32[10, 20], B: dace.float32[10, 20], C: dace.float32[10, 20], out: dace.float32[1]):
    A_prime = dace.define_local([10, 20], dtype=A.dtype)
    A_prime_copy = dace.define_local([10, 20], dtype=A.dtype)
    for i, j in dace.map[0:10, 0:20]:
        with dace.tasklet:
            inp << A[i, j]
            out1 >> out(1, lambda a, b: a + b)[0]
            out2 >> A_prime[i, j]
            out3 >> A_prime_copy[i, j]
            out1 = inp
            out2 = inp * inp
            out3 = inp * inp

    for i, j in dace.map[0:10, 0:20]:
        with dace.tasklet:
            inp << A_prime[i, j]
            out1 >> B[i, j]
            out1 = inp + 1

    for i, j in dace.map[0:10, 0:20]:
        with dace.tasklet:
            inp << A_prime_copy[i, j]
            out2 >> C[i, j]
            out2 = inp + 2


@dace.program
def fusion_with_transient(A: dace.float64[2, 20]):
    res = np.ndarray([2, 20], dace.float64)
    for i in dace.map[0:20]:
        for j in dace.map[0:2]:
            with dace.tasklet:
                a << A[j, i]
                t >> res[j, i]
                t = a * a
    for i in dace.map[0:20]:
        for j in dace.map[0:2]:
            with dace.tasklet:
                t << res[j, i]
                o >> A[j, i]
                o = t * 2


@fpga_test()
def test_multiple_fusions_fpga():
    sdfg = multiple_fusions.to_sdfg()
    sdfg.simplify()
    assert sdfg.apply_transformations_repeated(MapFusionVertical) >= 2
    assert sdfg.apply_transformations_repeated(FPGATransformSDFG) == 1
    A = np.random.rand(10, 20).astype(np.float32)
    B = np.zeros_like(A)
    C = np.zeros_like(A)
    out = np.zeros(shape=1, dtype=np.float32)
    sdfg(A=A, B=B, C=C, out=out)
    diff1 = np.linalg.norm(A * A + 1 - B)
    diff2 = np.linalg.norm(A * A + 2 - C)
    assert diff1 <= 1e-4
    assert diff2 <= 1e-4
    return sdfg


@fpga_test(assert_ii_1=False)
def test_fusion_with_transient_fpga():
    # To achieve II=1 with Xilinx, we need to decouple reads/writes from memory
    A = np.random.rand(2, 20)
    expected = A * A * 2
    sdfg = fusion_with_transient.to_sdfg()
    sdfg.simplify()
    assert sdfg.apply_transformations_repeated(MapFusionVertical) >= 2
    assert sdfg.apply_transformations_repeated(FPGATransformSDFG) == 1
    sdfg(A=A)
    assert np.allclose(A, expected)
    return sdfg


@xilinx_test(assert_ii_1=True)
def test_fusion_with_transient_fpga_decoupled():

    A = np.random.rand(2, 20)
    expected = A * A * 2
    sdfg = fusion_with_transient.to_sdfg()
    sdfg.simplify()
    assert sdfg.apply_transformations_repeated(MapFusionVertical) >= 2
    assert sdfg.apply_transformations_repeated(FPGATransformSDFG) == 1
    with set_temporary("compiler", "xilinx", "decouple_array_interfaces", value=True):
        sdfg(A=A)
    assert np.allclose(A, expected)
    return sdfg


if __name__ == "__main__":
    multiple_fusions_fpga(None)
    fusion_with_transient_fpga(None)
