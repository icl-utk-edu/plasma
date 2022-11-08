
#.rst:
# FindLAPACKE
# -----------
#
# Find LAPACKE header files and libraries
# This module sets the following variables:
#
# ::
#
#   LAPACKE_FOUND - set to true if a library implementing the BLAS C interface
#     is found

# if the caller didn't select LAPACKE then see if the environment variable has it
if (NOT LAPACKE_PROVIDER)
  if (NOT "x_$ENV{LAPACKE_PROVIDER}_x" STREQUAL "x__x")
    set(LAPACKE_PROVIDER $ENV{LAPACKE_PROVIDER})
  else ()
    set(LAPACKE_PROVIDER "any")
  endif()
endif()

if ("x_${LAPACKE_PROVIDER}_x" STREQUAL "x_generic_x" OR "x_${LAPACKE_PROVIDER}_x" STREQUAL "x_any_x")
    # the LAPACKE BLAS root path is defined, attempt to find the header and libraries
  if (LAPACKE_ROOT)
    find_path(LAPACKE_INCLUDE_DIRS lapacke.h PATHS ${LAPACKE_ROOT} ENV LAPACKE_ROOT PATH_SUFFIXES include DOC "Path to LapackE include directory")
    if (NOT LAPACKE_LIBRARIES)
      find_library(LAPACKE_LIBRARIES lapacke PATHS ${LAPACKE_ROOT} ENV LAPACKE_ROOT)
    endif()
  endif()
endif()

if ("x_${LAPACKE_PROVIDER}_x" STREQUAL "x_mkl_x" OR "x_${LAPACKE_PROVIDER}_x" STREQUAL "x_any_x")
  find_package( MKL )
  if (MKL_FOUND)
    set(LAPACKE_INCLUDE_DIRS ${MKL_INCLUDE_DIRS})
    set(LAPACKE_LIBRARIES ${MKL_LIBRARIES})
    set(LAPACKE_PROVIDER "mkl")
  endif()
endif()

if ("x_${LAPACKE_PROVIDER}_x" STREQUAL "x_openblas_x" OR "x_${LAPACKE_PROVIDER}_x" STREQUAL "x_any_x")
  find_package( OpenBLAS )
  if (OpenBLAS_FOUND)
    set(LAPACKE_INCLUDE_DIRS ${OpenBLAS_INCLUDE_DIRS})
    set(LAPACKE_LIBRARIES ${OpenBLAS_LIBRARIES})
    set(LAPACKE_PROVIDER "openblas")
  endif()
endif()

if ("x_${LAPACKE_PROVIDER}_x" STREQUAL "x_netlib_x" OR "x_${LAPACKE_PROVIDER}_x" STREQUAL "x_any_x")
  find_package( NetlibLapacke )
  if(NetlibLapacke_FOUND)
    set(LAPACKE_INCLUDE_DIRS ${NetlibLapacke_INCLUDE_DIRS})
    set(LAPACKE_LIBRARIES ${NetlibLapacke_LIBRARIES})
    set(LAPACKE_PROVIDER "netlib")
  endif()
endif()

find_package_handle_standard_args(LAPACKE REQUIRED_VARS LAPACKE_INCLUDE_DIRS LAPACKE_LIBRARIES LAPACKE_PROVIDER)
