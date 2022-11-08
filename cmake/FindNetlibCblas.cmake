
#.rst:
# FindNetlibCblas
# ---------------
#
# Find NetlibCblas library.
# This module sets the following variables:
#
# ::
#
#  NetlibCblas_FOUND - set to TRUE if a NetlibCblas library was found
#  NetlibCblas_INCLUDE_DIRS - Location of NetlibCblas header files
#  NetlibCblas_LINKER_FLAGS - linker flags required to link NetlibCblas
#  NetlibCblas_LIBRARIES - libraries required to link NetlibCblas
#
# User settings
# -------------
#
# NETLIB_ROOT is a directory that contains a NetlibCblas installation.
# ENV{NETLIB_ROOT} is an environment variable pointing to directory that contains a NetlibCblas installation.
#

if (NOT NetlibCblas_INCLUDE_DIRS)
  find_path(NetlibCblas_INCLUDE_DIRS cblas_mangling.h PATHS ${NETLIB_ROOT} ENV NETLIB_ROOT PATH_SUFFIXES include DOC "Path to Netlib C BLAS include directory")
endif()

if (NOT NetlibCblas_LIBRARIES)
  find_library(NetlibCblas_LIBRARIES cblas PATHS ${NETLIB_ROOT} ENV NETLIB_ROOT)
endif()

find_package_handle_standard_args(NetlibCblas REQUIRED_VARS NetlibCblas_INCLUDE_DIRS NetlibCblas_LIBRARIES)
