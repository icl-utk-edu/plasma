
#.rst:
# FindNetlibLapacke
# -----------------
#
# Find NetlibLapacke library.
# This module sets the following variables:
#
# ::
#
#  NetlibLapacke_FOUND - set to TRUE if a NetlibLapacke library was found
#  NetlibLapacke_INCLUDE_DIRS - Location of NetlibLapacke header files
#  NetlibLapacke_LINKER_FLAGS - linker flags required to link NetlibLapacke
#  NetlibLapacke_LIBRARIES - libraries required to link NetlibLapacke
#
# User settings
# -------------
#
# NETLIB_ROOT is a directory that contains a NetlibLapacke installation.
# ENV{NETLIB_ROOT} is an environment variable pointing to directory that contains a NetlibLapacke installation.
#

if (NOT NetlibLapacke_INCLUDE_DIRS)
  find_path(NetlibLapacke_INCLUDE_DIRS lapacke.h PATHS ${NETLIB_ROOT} ENV NETLIB_ROOT PATH_SUFFIXES include DOC "Path to Netlib LapackE include directory")
endif()

if (NOT NetlibLapacke_LIBRARIES)
  find_library(NetlibLapacke_LIBRARIES lapacke PATHS ${NETLIB_ROOT} ENV NETLIB_ROOT)
endif()

find_package_handle_standard_args(NetlibLapacke REQUIRED_VARS NetlibLapacke_INCLUDE_DIRS NetlibLapacke_LIBRARIES)
