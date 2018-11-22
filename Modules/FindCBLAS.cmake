
#.rst:
# FindCBLAS
# ---------
#
# Find CBLAS header files and libraries
# This module sets the following variables:
#
# ::
#
#   CBLAS_FOUND - set to true if a library implementing the BLAS C interface
#     is found

# if the caller didn't select CBLAS then see if the environment variable has it
if (NOT CBLAS_PROVIDER)
  if (NOT "x_$ENV{CBLAS_PROVIDER}_x" STREQUAL "x__x")
    set(CBLAS_PROVIDER $ENV{CBLAS_PROVIDER})
  else ()
    set(CBLAS_PROVIDER "any")
  endif()
endif()

if ("x_${CBLAS_PROVIDER}_x" STREQUAL "x_generic_x" OR "x_${CBLAS_PROVIDER}_x" STREQUAL "x_any_x")
  # the C BLAS root path is defined, attempt to find the header and libraries
  if (CBLAS_ROOT)
    find_path(CBLAS_INCLUDE_DIRS cblas.h PATHS ${CBLAS_ROOT} ENV CBLAS_ROOT PATH_SUFFIXES include DOC "Path to C BLAS include directory")
    if (NOT CBLAS_LIBRARIES)
      find_library(CBLAS_LIBRARIES cblas PATHS ${CBLAS_ROOT} ENV CBLAS_ROOT)
    endif()
  endif()
endif()

if ("x_${CBLAS_PROVIDER}_x" STREQUAL "x_mkl_x" OR "x_${CBLAS_PROVIDER}_x" STREQUAL "x_any_x")
  find_package( MKL )
  if (MKL_FOUND)
    set(CBLAS_INCLUDE_DIRS ${MKL_INCLUDE_DIRS})
    set(CBLAS_LIBRARIES ${MKL_LIBRARIES})
    set(CBLAS_PROVIDER "mkl")
  endif()
endif()

if ("x_${CBLAS_PROVIDER}_x" STREQUAL "x_openblas_x" OR "x_${CBLAS_PROVIDER}_x" STREQUAL "x_any_x")
  find_package( OpenBLAS )
  if (OpenBLAS_FOUND)
    set(CBLAS_INCLUDE_DIRS ${OpenBLAS_INCLUDE_DIRS})
    set(CBLAS_LIBRARIES ${OpenBLAS_LIBRARIES})
    set(CBLAS_PROVIDER "openblas")
  endif()
endif()

if ("x_${CBLAS_PROVIDER}_x" STREQUAL "x_netlib_x" OR "x_${CBLAS_PROVIDER}_x" STREQUAL "x_any_x")
  find_package( NetlibCblas )
  if (NetlibCblas_FOUND)
    set(CBLAS_INCLUDE_DIRS ${NetlibCblas_INCLUDE_DIRS})
    set(CBLAS_LIBRARIES ${NetlibCblas_LIBRARIES})
    set(CBLAS_PROVIDER "netlib")
  endif()
endif()

find_package_handle_standard_args(CBLAS REQUIRED_VARS CBLAS_INCLUDE_DIRS CBLAS_LIBRARIES CBLAS_PROVIDER)
