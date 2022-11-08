
#.rst:
# FindOpenBLAS
# ---------
#
# Find OpenBLAS library.
# This module sets the following variables:
#
# ::
#
#  OpenBLAS_FOUND - set to TRUE if a OpenBLAS library was found
#  OpenBLAS_INCLUDE_DIRS - Location of OpenBLAS header files
#  OpenBLAS_LINKER_FLAGS - linker flags required to link OpenBLAS
#  OpenBLAS_LIBRARIES - libraries required to link OpenBLAS
#
# User settings
# -------------
#
# OpenBLAS_ROOT is a directory that contains a OpenBLAS installation.
# ENV{OPENBLAS_ROOT} is an environment variable pointing to directory that contains a OpenBLAS installation.
#

#FIXME: lib/cmake/openblas/OpenBLASConfig.cmake

if (NOT OpenBLAS_INCLUDE_DIRS)
  find_path(OpenBLAS_INCLUDE_DIRS openblas_config.h PATHS ${OpenBLAS_ROOT} ENV OPENBLAS_ROOT PATH_SUFFIXES include DOC "Path to OpenBLAS include directory")
endif()

if (NOT OpenBLAS_LIBRARIES)
  find_library(OpenBLAS_LIBRARIES openblas PATHS ${OpenBLAS_ROOT} ENV OPENBLAS_ROOT)
endif()

find_package_handle_standard_args(OpenBLAS REQUIRED_VARS OpenBLAS_INCLUDE_DIRS OpenBLAS_LIBRARIES)
