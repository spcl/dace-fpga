# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from dace_fpga.fpga_testing import fpga_test
from dace_fpga.transformations import FPGATransformSDFG

# Define symbols for output size
W = dace.symbol("W")
H = dace.symbol("H")


@dace.program
def mandelbrot(output: dace.uint16[H, W], maxiter: dace.int64):
    for py, px in dace.map[0:H, 0:W]:
        x0 = -2.5 + ((float(px) / W) * 3.5)
        y0 = -1 + ((float(py) / H) * 2)
        x = 0.0
        y = 0.0
        iteration = 0
        while (x * x + y * y < 2 * 2 and iteration < maxiter):
            xtemp = x * x - y * y + x0
            y = 2 * x * y + y0
            x = xtemp
            iteration = iteration + 1

        output[py, px] = iteration


# TODO: Pipeline control flow while-loop?
@fpga_test(assert_ii_1=False)
def test_mandelbrot_fpga():
    h, w, max_iterations = 64, 64, 1000
    out = dace.ndarray([h, w], dtype=dace.uint16)
    out[:] = dace.uint32(0)
    sdfg = mandelbrot.to_sdfg()
    sdfg.apply_transformations(FPGATransformSDFG)
    sdfg(output=out, maxiter=max_iterations, W=w, H=h)
    return sdfg


if __name__ == "__main__":
    test_mandelbrot_fpga(None)
