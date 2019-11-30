#.rst:
# FindAccelerate
# ---------------
#
# Find Apple's Accelerate library.
# This module sets the following variables:
#
# ::
#
#  Accelerate_FOUND - set to TRUE if a Accelerate library was found
#  Accelerate_INCLUDE_DIRS - Location of Accelerate header files
#  Accelerate_LINKER_FLAGS - linker flags required to link Accelerate
#  Accelerate_LIBRARIES - libraries required to link Accelerate
#
# User settings
# -------------
#
# ACCELERATE_ROOT is a directory that contains a Accelerate installation.
# ENV{ACCELERATE_ROOT} is an environment variable pointing to directory that contains a Accelerate installation.
#

if (NOT Accelerate_INCLUDE_DIRS)
  find_path(Accelerate_INCLUDE_DIRS Accelerate/Accelerate.h PATHS ${ACCELERATE_ROOT} ENV ACCELERATE_ROOT PATH_SUFFIXES include DOC "Path to Apple's Accelerate include directory")
endif()

if (NOT Accelerate_LIBRARIES)
  find_package(BLAS REQUIRED)

  # add BLAS libraries to CMake's link line (Accelerate is searched and found by CMake's FindBLAS.cmake module)
  set(CMAKE_REQUIRED_LIBRARIES ${BLAS_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${MATH_LIB})

  include(CheckSymbolExists)
  check_symbol_exists(cblas_cgemm Accelerate/Accelerate.h Accelerate_WORKS)
  if (Accelerate_WORKS)
    set(Accelerate_LIBRARIES ${BLAS_LIBRARIES})
  endif()
endif()

find_package_handle_standard_args(Accelerate REQUIRED_VARS Accelerate_INCLUDE_DIRS Accelerate_LIBRARIES)
