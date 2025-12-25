// Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
#ifndef __DACE_FPGA_RUNTIME_H
#define __DACE_FPGA_RUNTIME_H

#ifdef DACE_XILINX
#include "xilinx/host.h"
#include "xilinx/vec.h"
#endif

#ifdef DACE_INTELFPGA
#include "intel_fpga/host.h"
#endif

#include "fpga_common.h"

#endif // __DACE_FPGA_RUNTIME_H
