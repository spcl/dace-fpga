
set(DACE_RTLLIB_DIR ${CMAKE_CURRENT_LIST_DIR}/../external/rtllib)
include ("${DACE_RTLLIB_DIR}/cmake/rtl_target.cmake")

# create verilator RTL simulation objects
if (DACE_ENABLE_XILINX AND (NOT (DACE_XILINX_MODE STREQUAL "simulation")))
  # Get all of the kernel names
  list(APPEND RTL_KERNELS "")
  foreach(RTL_FILE ${DACE_RTL_FILES})
      get_filename_component(RTL_KERNEL ${RTL_FILE} DIRECTORY)
      list(APPEND RTL_KERNELS ${RTL_KERNEL})
  endforeach()
  list(REMOVE_DUPLICATES RTL_KERNELS)

  # Prepare build folders
  set (RTL_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/rtl/generated")
  set (RTL_LOG_DIR       "${CMAKE_CURRENT_BINARY_DIR}/rtl/log")
  set (RTL_TEMP_DIR      "${CMAKE_CURRENT_BINARY_DIR}/rtl/tmp")
  file (MAKE_DIRECTORY
      ${RTL_GENERATED_DIR}
      ${RTL_LOG_DIR}
      ${RTL_TEMP_DIR})
  execute_process(COMMAND ${Vitis_PLATFORMINFO} -p ${DACE_XILINX_TARGET_PLATFORM} -jhardwarePlatform.board.part
    OUTPUT_VARIABLE RTL_PART
    RESULT_VARIABLE _platforminfo_res)

  if (NOT ${_platforminfo_res} EQUAL 0)
    message(FATAL_ERROR "No part was found for platform ${DACE_XILINX_TARGET_PLATFORM} after querying 'platforminfo -p ${DACE_XILINX_TARGET_PLATFORM} -j\"hardwarePlatform.board.part\"'")
  endif()

  # Generate all of the .xo targets
  foreach(RTL_SRC_DIR ${RTL_KERNELS})
      get_filename_component(RTL_KERNEL ${RTL_SRC_DIR} NAME)
      get_filename_component(RTL_SCRIPTS "${RTL_SRC_DIR}/../scripts" ABSOLUTE)
      set(RTL_XO "${RTL_KERNEL}.xo")
      rtllib_rtl_target(${RTL_KERNEL} ${RTL_SRC_DIR} ${RTL_SCRIPTS} ${RTL_GENERATED_DIR} ${RTL_LOG_DIR} ${RTL_TEMP_DIR} "${RTLLIB_DIR}/rtl" ${RTL_XO} ${RTL_PART} "" "\"\"")
      add_custom_target(${RTL_KERNEL} DEPENDS ${RTL_XO})
      set(DACE_RTL_KERNELS ${DACE_RTL_KERNELS} ${RTL_XO})
      set(DACE_RTL_DEPENDS ${DACE_RTL_DEPENDS} ${RTL_KERNEL})
  endforeach()
else()
  # find verilator installation
  find_package(verilator HINTS $ENV{VERILATOR_ROOT} ${VERILATOR_ROOT})
  if (NOT verilator_FOUND)
    message(FATAL_ERROR "Verilator was not found. Either install it, or set the VERILATOR_ROOT environment variable")
  endif()

  # check minimal version requirements
  set(VERILATOR_MIN_VERSION "4.028")
  if("${verilator_VERSION}" VERSION_LESS VERILATOR_MIN_VERSION)
    message(ERROR "Please upgrade verilator to version >=${VERILATOR_MIN_VERSION}")
  endif()

  # get verilator flags from dace.conf
  set(VERILATOR_FLAGS "${DACE_RTL_VERILATOR_FLAGS}")

  # add lint verilator flags
  if("${DACE_RTL_VERILATOR_LINT_WARNINGS}")
    # -Wall: Enable all style warnings
    # -Wno-fatal: Disable fatal exit on warnings
    set(VERILATOR_FLAGS "${VERILATOR_FLAGS}" "-Wall" "-Wno-fatal")
  endif()

  # add verilated.cpp source
  set(DACE_CPP_FILES "${DACE_CPP_FILES}" "${VERILATOR_ROOT}/include/verilated.cpp" "${VERILATOR_ROOT}/include/verilated_threads.cpp" )

  foreach(RTL_FILE ${DACE_RTL_FILES})

    # extract design name
    get_filename_component(RTL_FILE_NAME "${RTL_FILE}" NAME_WE)

    # add verilated design
    add_library("${RTL_FILE_NAME}" OBJECT)

    # include verilator
    set(VERILATOR_INCLUDE "${VERILATOR_ROOT}/include" "${dace_program_BINARY_DIR}/CMakeFiles/${RTL_FILE_NAME}.dir/V${RTL_FILE_NAME}.dir")
    include_directories(${VERILATOR_INCLUDE})

    # verilate design
    verilate("${RTL_FILE_NAME}" SOURCES ${RTL_FILE} VERILATOR_ARGS "${VERILATOR_FLAGS}")
    file(GLOB VSRC_FILES "${dace_program_BINARY_DIR}/CMakeFiles/${RTL_FILE_NAME}.dir/V${RTL_FILE_NAME}.dir/*.cpp")
    set(DACE_CPP_FILES "${DACE_CPP_FILES}" ${VSRC_FILES} "${dace_program_BINARY_DIR}/CMakeFiles/${RTL_FILE_NAME}.dir/V${RTL_FILE_NAME}.dir/V${RTL_FILE_NAME}.cpp")

    # add object library for linking
    set(DACE_LIBS ${DACE_LIBS} ${${RTL_FILE_NAME}})

  endforeach()
endif()
