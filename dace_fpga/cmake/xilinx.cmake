set(HLSLIB_PART_NAME "${DACE_XILINX_PART_NAME}")

# Allow passing flags to various stages of Xilinx compilation process
set(DACE_XILINX_MODE "simulation" CACHE STRING "Type of compilation/execution [simulation/software_emulation/hardware_emulation/hardware].")
set(DACE_XILINX_HOST_FLAGS "" CACHE STRING "Extra flags to host code")
set(DACE_XILINX_SYNTHESIS_FLAGS "" CACHE STRING "Extra flags for performing high-level synthesis")
set(DACE_XILINX_BUILD_FLAGS "" CACHE STRING "Extra flags to xocc build phase")
set(DACE_XILINX_TARGET_CLOCK "" CACHE STRING "Target clock frequency of FPGA kernel")
set(DACE_XILINX_PART_NAME "xcu280-fsvh2892-2L-e" CACHE STRING "Xilinx chip to target from HLS")
set(DACE_XILINX_TARGET_PLATFORM "xilinx_u280_xdma_201920_1" CACHE STRING "Vitis platform to target")
set(DACE_XILINX_ENABLE_DEBUGGING OFF CACHE STRING "Inject debugging cores to kernel build (always on for simulation/emulation)")

# Find Vitis installation
find_package(Vitis REQUIRED)
include_directories(SYSTEM ${Vitis_INCLUDE_DIRS})
add_definitions(-DDACE_XILINX -DDACE_VITIS_DIR=\"${VITIS_ROOT_DIR}\")
set(DACE_LIBS ${DACE_LIBS} ${Vitis_LIBRARIES})

# Create Xilinx object files
if (DACE_XILINX_TARGET_CLOCK MATCHES "[|]")
  string(REGEX MATCH "0:([0-9]+)" DACE_XILINX_EXTERNAL_TARGET_CLOCK ${DACE_XILINX_TARGET_CLOCK})
  string(REGEX MATCH "1:([0-9]+)" DACE_XILINX_INTERNAL_TARGET_CLOCK ${DACE_XILINX_TARGET_CLOCK})
  string(SUBSTRING ${DACE_XILINX_EXTERNAL_TARGET_CLOCK} 2 -1 DACE_XILINX_EXTERNAL_TARGET_CLOCK)
  string(SUBSTRING ${DACE_XILINX_INTERNAL_TARGET_CLOCK} 2 -1 DACE_XILINX_INTERNAL_TARGET_CLOCK)
else()
  set(DACE_XILINX_EXTERNAL_TARGET_CLOCK ${DACE_XILINX_TARGET_CLOCK})
  set(DACE_XILINX_INTERNAL_TARGET_CLOCK ${DACE_XILINX_TARGET_CLOCK})
endif()

if((NOT (DACE_XILINX_MODE STREQUAL "hardware")) OR DACE_XILINX_ENABLE_DEBUGGING)
  set(DACE_XILINX_HOST_FLAGS "${DACE_XILINX_HOST_FLAGS} -g")
endif()

set_source_files_properties(${DACE_XILINX_KERNEL_FILES} ${DACE_XILINX_HOST_FILES} PROPERTIES COMPILE_FLAGS "${DACE_XILINX_HOST_FLAGS}")
set_source_files_properties(${DACE_XILINX_KERNEL_FILES} PROPERTIES COMPILE_FLAGS "-DDACE_XILINX_DEVICE_CODE ${DACE_XILINX_HOST_FLAGS}")
set(DACE_OBJECTS ${DACE_OBJECTS} ${DACE_XILINX_KERNEL_FILES} ${DACE_XILINX_HOST_FILES})

if(DACE_XILINX_MODE STREQUAL "simulation")
  # This will cause the OpenCL calls to instead call a simulation code
  # running on the host
  add_definitions(-DHLSLIB_SIMULATE_OPENCL)
endif()

if(DACE_MINIMUM_FIFO_DEPTH)
  set(DACE_XILINX_MINIMUM_FIFO_DEPTH "\nconfig_dataflow -fifo_depth ${DACE_MINIMUM_FIFO_DEPTH}")
endif()


# If the project uses generated IP cores (e.g. through multi-pumping)
if(DACE_XILINX_IP_FILES)
  set(DACE_XILINX_BUILD_FLAGS ${DACE_XILINX_BUILD_FLAGS} --user_ip_repo_paths ip_cores)
endif()

unset(DACE_KERNEL_TARGETS)

# Generate the target kernel for each IP (multi-pumped kernel)
foreach(DACE_IP ${DACE_XILINX_IP_FILES})
  get_filename_component(DACE_KERNEL_NAME ${DACE_IP} NAME_WE)
  get_filename_component(DACE_KERNEL_SRC ${DACE_IP} DIRECTORY)

  # Configure the tcl script for packaging the C++ kernel as an IP core for Vivado.
  configure_file(${CMAKE_CURRENT_LIST_DIR}/Xilinx_IP.tcl.in Package_${DACE_KERNEL_NAME}.tcl)
  add_custom_command(
    OUTPUT ip_cores/${DACE_KERNEL_NAME}/impl/export.zip
    COMMAND XILINX_PATH=${CMAKE_BINARY_DIR} ${Vitis_HLS}
    -f Package_${DACE_KERNEL_NAME}.tcl
    DEPENDS ${DACE_IP}
  )

  # Get the hardware part of the board, which is needed to package the .xo file.
  execute_process(COMMAND ${Vitis_PLATFORMINFO} -p ${DACE_XILINX_TARGET_PLATFORM} -jhardwarePlatform.board.part
  OUTPUT_VARIABLE RTL_PART
  RESULT_VARIABLE _platforminfo_res
  OUTPUT_STRIP_TRAILING_WHITESPACE)

  # Add target for packaging the kernel into an .xo file.
  set (RTL_XO "${DACE_KERNEL_NAME}.xo")
  rtllib_rtl_target(${DACE_KERNEL_NAME} ${DACE_KERNEL_SRC} ${DACE_KERNEL_SRC} ${DACE_KERNEL_SRC} log tmp ${DACE_KERNEL_SRC} ${RTL_XO} ${RTL_PART} ip_cores/${DACE_KERNEL_NAME}/impl/export.zip ip_cores)
  add_custom_target(${DACE_KERNEL_NAME} DEPENDS ${RTL_XO})
  set(DACE_RTL_KERNELS ${DACE_RTL_KERNELS} ${RTL_XO})
  set(DACE_RTL_DEPENDS ${DACE_RTL_DEPENDS} ${DACE_KERNEL_NAME})
endforeach()

foreach(DACE_KERNEL_FILE ${DACE_XILINX_KERNEL_FILES})
  # Extract kernel name
  get_filename_component(DACE_KERNEL_NAME ${DACE_KERNEL_FILE} NAME)
  string(REGEX REPLACE "(.+).cpp" "\\1" DACE_KERNEL_NAME "${DACE_KERNEL_NAME}")

  add_vitis_kernel(${DACE_KERNEL_NAME}
                    FILES ${DACE_VITIS_KERNEL_FILES} ${DACE_KERNEL_FILE}
                    HLS_FLAGS "${DACE_XILINX_SYNTHESIS_FLAGS} -DDACE_SYNTHESIS -DDACE_XILINX -DDACE_XILINX_DEVICE_CODE"
                    HLS_CONFIG "config_compile -pipeline_style frp${DACE_XILINX_MINIMUM_FIFO_DEPTH}"
                    INCLUDE_DIRS ${CMAKE_CURRENT_LIST_DIR}/../external/hlslib/include
                                 ${CMAKE_SOURCE_DIR}/../runtime/include
                                 ${CMAKE_CURRENT_LIST_DIR}/../runtime/include)
  set(DACE_KERNEL_TARGETS ${DACE_KERNEL_TARGETS} ${DACE_KERNEL_NAME})
endforeach()

add_vitis_program(${DACE_PROGRAM_NAME}
                  ${DACE_XILINX_TARGET_PLATFORM}
                  KERNELS ${DACE_KERNEL_TARGETS}
                  DEBUGGING ${DACE_XILINX_ENABLE_DEBUGGING}
                  CLOCK ${DACE_XILINX_EXTERNAL_TARGET_CLOCK}
                  BUILD_FLAGS ${DACE_XILINX_BUILD_FLAGS}
                  LINK_FLAGS ${DACE_RTL_KERNELS}
                  DEPENDS ${DACE_RTL_DEPENDS}
                  CONFIG ${DACE_XILINX_CONFIG_FILE})

# Add additional required files
if(DACE_XILINX_MODE STREQUAL "software_emulation" AND DACE_FPGA_AUTOBUILD_BITSTREAM)
  add_custom_target(autobuild_bitstream ALL
                    COMMENT "Automatically built bitstream for software emulation."
                    DEPENDS sw_emu)
endif()
if(DACE_XILINX_MODE STREQUAL "hardware_emulation" AND DACE_FPGA_AUTOBUILD_BITSTREAM)
  add_custom_target(autobuild_bitstream ALL
                    COMMENT "Automatically built bitstream for hardware emulation."
                    DEPENDS hw_emu)
endif()
if(DACE_XILINX_MODE STREQUAL "hardware" AND DACE_FPGA_AUTOBUILD_BITSTREAM)
  add_custom_target(autobuild_bitstream ALL
                    COMMENT "Automatically built bitstream for hardware."
                    DEPENDS hw)
endif()
