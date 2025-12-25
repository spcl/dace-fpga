
set(DACE_INTELFPGA_MODE "simulation" CACHE STRING "Type of compilation/execution [emulator/simulator/hardare].")
set(DACE_INTELFPGA_HOST_FLAGS "" CACHE STRING "Extra flags to host compiler.")
set(DACE_INTELFPGA_KERNEL_FLAGS "" CACHE STRING "Extra flags to kernel compiler.")
set(DACE_INTELFPGA_TARGET_BOARD "a10gx" CACHE STRING "Target FPGA board.")
set(DACE_INTELFPGA_ENABLE_DEBUGGING OFF CACHE STRING "Enable debugging.")


find_package(IntelFPGAOpenCL REQUIRED)
include_directories(SYSTEM ${IntelFPGAOpenCL_INCLUDE_DIRS})
add_definitions(-DDACE_INTELFPGA)
set(DACE_LIBS ${DACE_LIBS} ${IntelFPGAOpenCL_LIBRARIES})

# Create Intel FPGA object files
if((NOT (DACE_INTELFPGA_MODE STREQUAL "hardware")) OR DACE_INTELFPGA_ENABLE_DEBUGGING)
  set(DACE_INTELFPGA_HOST_FLAGS "${DACE_INTELFPGA_HOST_FLAGS} -g")
  set(DACE_INTELFPGA_SYNTHESIS_FLAGS "${DACE_INTELFPGA_KERNEL_FLAGS} -fast-compile -profile=all -g -fast-emulator")
endif()

set_source_files_properties(${DACE_INTELFPGA_KERNEL_FILES} ${DACE_INTELFPGA_HOST_FILES} PROPERTIES COMPILE_FLAGS "${DACE_INTELFPGA_HOST_FLAGS}")
set_source_files_properties(${DACE_INTELFPGA_KERNEL_FILES} PROPERTIES COMPILE_FLAGS "-DDACE_INTELFPGA_DEVICE_CODE ${DACE_INTELFPGA_HOST_FLAGS}")
set(DACE_OBJECTS ${DACE_OBJECTS} ${DACE_INTELFPGA_KERNEL_FILES} ${DACE_INTELFPGA_HOST_FILES})

# Add synthesis and build commands
set(DACE_AOC_KERNEL_FILES)
set(DACE_AOC_DEFINITIONS "-DDACE_INTELFPGA")
foreach(DACE_KERNEL_FILE ${DACE_INTELFPGA_KERNEL_FILES})

  get_filename_component(DACE_KERNEL_NAME ${DACE_KERNEL_FILE} NAME)
  string(REGEX REPLACE "kernel_(.+).cl" "\\1" DACE_KERNEL_NAME "${DACE_KERNEL_NAME}")
  set(DACE_AOC_KERNEL_FILES ${DACE_AOC_KERNEL_FILES} ${DACE_KERNEL_FILE})

  # Intel compiler does not allow to specify the output file if more than input file is used.
  # In this case, the output AOCX file will be named as the last OpenCL file given in input to the compiler.
  # We need to save the name of the last input file, so that later we can assign a proper name to the produced bitstream.
  get_filename_component(DACE_AOC_OUTPUT_FILE ${DACE_KERNEL_FILE} NAME_WE)
endforeach()

string(REPLACE " " ";" DACE_INTELFPGA_KERNEL_FLAGS_INTERNAL
        "${DACE_INTELFPGA_KERNEL_FLAGS}")

set(DACE_AOC_BUILD_FLAGS
  -I${CMAKE_CURRENT_LIST_DIR}/../external/hlslib/include
  -I${CMAKE_SOURCE_DIR}/../runtime/include
  -I${CMAKE_CURRENT_LIST_DIR}/../runtime/include
  -I${CMAKE_BINARY_DIR}
  -board=${DACE_INTELFPGA_TARGET_BOARD}
  ${DACE_INTELFPGA_KERNEL_FLAGS_INTERNAL}
  ${DACE_AOC_DEFINITIONS})

add_custom_target(
  intelfpga_report_${DACE_PROGRAM_NAME}
  COMMAND
  ${IntelFPGAOpenCL_AOC}
  ${DACE_AOC_BUILD_FLAGS}
  ${DACE_AOC_KERNEL_FILES}
  -rtl
  -report
  COMMAND mv ${DACE_AOC_OUTPUT_FILE} ${DACE_PROGRAM_NAME})

add_custom_command(
  OUTPUT ${DACE_PROGRAM_NAME}_emulator.aocx
  COMMAND ${IntelFPGAOpenCL_AOC}
  ${DACE_AOC_BUILD_FLAGS}
  -march=emulator
  ${DACE_AOC_KERNEL_FILES}
  COMMAND mv ${DACE_AOC_OUTPUT_FILE}.aocx  ${DACE_PROGRAM_NAME}_emulator.aocx
  DEPENDS ${DACE_AOC_KERNEL_FILES})

add_custom_command(
  OUTPUT ${DACE_PROGRAM_NAME}_hardware.aocx
  COMMAND ${IntelFPGAOpenCL_AOC}
  ${DACE_AOC_BUILD_FLAGS}
  ${DACE_AOC_KERNEL_FILES}
  COMMAND mv ${DACE_AOC_OUTPUT_FILE}.aocx  ${DACE_PROGRAM_NAME}_hardware.aocx
  COMMAND mv ${DACE_AOC_OUTPUT_FILE} ${DACE_PROGRAM_NAME}
  DEPENDS ${DACE_AOC_KERNEL_FILES})


# Add additional required files
if(DACE_INTELFPGA_MODE STREQUAL "emulator")
    add_custom_target(intelfpga_compile_${DACE_PROGRAM_NAME}_emulator
                      ALL DEPENDS ${DACE_PROGRAM_NAME}_emulator.aocx)
else()
    add_custom_target(intelfpga_compile_${DACE_PROGRAM_NAME}_emulator
                      DEPENDS ${DACE_PROGRAM_NAME}_emulator.aocx)
endif()
if(DACE_INTELFPGA_MODE STREQUAL "hardware" AND DACE_FPGA_AUTOBUILD_BITSTREAM)
    add_custom_target(intelfpga_compile_${DACE_PROGRAM_NAME}_hardware
                      ALL DEPENDS ${DACE_PROGRAM_NAME}_hardware.aocx)
else()
    add_custom_target(intelfpga_compile_${DACE_PROGRAM_NAME}_hardware
                      DEPENDS ${DACE_PROGRAM_NAME}_hardware.aocx)
endif()
