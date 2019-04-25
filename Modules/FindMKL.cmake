
#.rst:
# FindMKL
# -------
#
# Find header files for Intel MKL installation.
# This module sets the following variables:
#
# ::
#
#  MKL_FOUND - set to true if a library implementing the BLAS C interface
#     is found
#  MKL_FOUND - set to true if MKL implementatiom of C BLAS was found
#  MKL_INCLUDE_DIRS - location of MKL header files
#  MKL_LIBRARIES - libraries required to link for MKL's C interface to BLAS
#
# User settings
# -------------
#
# MKLROOT a directory that contains MKL installation.
# ENV{MKLROOT} a directory that contains MKL installation.
#

if (NOT MKL_INCLUDE_DIRS)
  find_path(MKL_INCLUDE_DIRS mkl.h PATHS ${MKLROOT} ENV MKLROOT PATH_SUFFIXES include DOC "Path to MKL include directory")
endif()

if (NOT MKL_LIBRARIES)
  #find_library(MKL_LIBRARIES mkl_core NAMES mkl_sequential PATHS MKLROOT ENV MKLROOT)
  find_package(BLAS REQUIRED)
  set(CMAKE_REQUIRED_INCLUDES ${MKL_INCLUDE_DIRS})
  if (NOT WIN32)
      set(MATH_LIB "-lm")
   endif ()
  set(CMAKE_REQUIRED_LIBRARIES ${BLAS_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${MATH_LIB})
  check_symbol_exists(cblas_cgemm mkl.h MKL_WORKS)
  unset(CMAKE_REQUIRED_INCLUDES)
  unset(CMAKE_REQUIRED_LIBRARIES)

  if (MKL_WORKS)
    set(MKL_LIBRARIES ${BLAS_LIBRARIES})
  endif()
endif()

find_package_handle_standard_args(MKL REQUIRED_VARS MKL_INCLUDE_DIRS MKL_LIBRARIES)
