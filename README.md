![FPGA CI](https://github.com/spcl/dace-fpga/actions/workflows/fpga-ci.yml/badge.svg)

DaCe-FPGA - FPGA Backend for DaCe
=================================

FPGA code generation support for the [DaCe](https://github.com/spcl/dace) data-centric framework. DaCe-FPGA adds
FPGA-specific types, transformations, code generation backends, and library node implementations to produce efficient
Xilinx and Intel FPGA designs directly from DaCe programs.

Quick Start
-----------

1. Install DaCe: `pip install dace`.
2. Clone this repository and fetch submodules: `git clone https://github.com/spcl/dace-fpga.git && cd dace-fpga && git submodule update --init --recursive`.
3. Make the package importable in the same environment as DaCe, for example: `export PYTHONPATH=$(pwd):$PYTHONPATH`.
4. Run a sample to verify the toolchain (requires vendor tools or emulation): `python samples/gemv_fpga.py`.

Repository Contents
-------------------

- [dace_fpga](dace_fpga): FPGA-specific types, transformations, code generation backends, and runtime headers.
- [samples](samples): end-to-end FPGA examples (e.g., GEMV, AXPY, systolic GEMM, histogram, streaming SpMV) and RTL integration samples.
- [tests](tests): unit and regression tests for FPGA compilation flows and transformations.
- [docs](docs): notes on FPGA code generation and optimization.
- [dace_fpga/external](dace_fpga/external): external dependencies (hlslib and rtllib), fetched via git submodules.

Dependencies
------------

- Xilinx FPGAs: Vitis HLS v2020.x or v2021.x (validated on u250 and u280 devices).
- Intel FPGAs: Intel FPGA SDK for OpenCL Pro edition v18.1 or v19.1 (validated on Arria 10 and Stratix 10 devices).
- A working DaCe installation and a supported Python 3 environment with CMake and a C++ toolchain.

Using DaCe-FPGA
---------------

- Write or generate an SDFG with DaCe as usual and target FPGA using the DaCe configuration and transformations (e.g., `FPGATransformSDFG`, streaming and memory layout passes).
- Use FPGA-specialized library node expansions (GEMV, GEMM, dot, reductions, etc.) provided under `dace_fpga` to achieve initiation-interval-1 pipelines.
- Customize code generation and hardware packages through the backends in [dace_fpga/codegen](dace_fpga/codegen) and runtime headers in [dace_fpga/runtime](dace_fpga/runtime).
- See additional guidance on how to optimize DaCe programs for FPGAs in [docs/optimization.rst](docs/optimization.rst) and the [DaCe documentation](https://spcldace.readthedocs.io).

Testing
-------

- Run the FPGA regression suite with `pytest tests`. Some tests require access to vendor toolchains or emulation devices; without them, skip or mark the relevant cases using pytest selectors.

Troubleshooting
---------------

- **Intel FPGA libraries not found**: when targeting Intel FPGAs, CMake may report `Could NOT find IntelFPGAOpenCL` because the Intel OpenCL compiler returns an include path that lacks the OpenCL host headers. DaCe relies on `hlslib`, which in turn uses the compiler-reported include path. Verify that the path returned by `aocl compile-config` contains `cl.hpp` and `cl2.hpp`. If it does not, locate these headers under the Intel Quartus installation and symlink or copy them into the path reported by `aocl`.

Contributing
------------

Contributions are welcome! Please follow the community guidelines in [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) and submit changes as pull requests.

License
-------

DaCe and its FPGA backend are released under the BSD 3-Clause License. See [LICENSE](LICENSE) for details.
