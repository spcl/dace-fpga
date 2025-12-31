# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import os

from dace.config import Config
from dace import dtypes
from dace.codegen import dispatcher
from dace.sdfg.type_inference import KNOWN_FUNCTIONS

# Extend the configuration schema with FPGA-specific settings
Config.extend(os.path.join(os.path.dirname(__file__), 'fpga_config_schema.yml'))

# Extend DaCe data types with FPGA-specific types
dtypes.ScheduleType.register('FPGA_Multi_Pumped')  #: Used for double pumping
dtypes.SCOPEDEFAULT_SCHEDULE[dtypes.ScheduleType.FPGA_Multi_Pumped] = dtypes.ScheduleType.FPGA_Device

# dtypes.StorageType.register('FPGA_ShiftRegister')  #: Only accessible at constant indices
# dtypes.FPGA_STORAGES.append(dtypes.StorageType.FPGA_ShiftRegister)

dtypes.ScheduleType.register('Unrolled')
dtypes.SCOPEDEFAULT_SCHEDULE[dtypes.ScheduleType.Unrolled] = dtypes.ScheduleType.CPU_Multicore

dtypes.InstrumentationType.register('FPGA')

# Register FPGA code generation targets and DefinedTypes
dispatcher.DefinedType.register('FPGA_ShiftRegister')  # A shift-register object used in FPGA code generation
from dace_fpga.codegen import intel_fpga, rtl, unroller, xilinx
from dace_fpga import instrumentation

# Register FPGA-specific transformations and passes
from dace_fpga import transformations

# Register library node expansions
from dace_fpga import library_nodes

# Register type inference functions

# Reading from an Intel FPGA channel returns the channel type
KNOWN_FUNCTIONS['read_channel_intel'] = lambda arg_types: arg_types[0]
