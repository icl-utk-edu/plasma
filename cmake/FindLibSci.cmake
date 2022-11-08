#.rst:
# FindLibSci
# -------
#
# Find header files for Cray LibSci installation.
# This module sets the following variables:
#
# ::
#
#  LibSci_FOUND - set to true if LibSci implementatiom of CBLAS was found
#  LibSci_INCLUDE_DIRS - location of LibSci CBLAS header files
#  LibSci_LIBRARIES - libraries required to link for C interface to BLAS
#
# User settings
# -------------
#
# LibSci_ROOT a directory that contains LibSci CBLAS installation.
# ENV{LibSci_ROOT} a directory that contains LibSci CBLAS installation.
#

if (NOT LibSci_INCLUDE_DIRS)
  find_path(LibSci_INCLUDE_DIRS cblas.h PATHS ${LibSci_ROOT} ENV LibSci_ROOT PATH_SUFFIXES include DOC "Path to LibSci CBLAS include directory")
  if (LibSci_INCLUDE_DIRS)
    set(CMAKE_REQUIRED_INCLUDES ${LibSci_INCLUDE_DIRS})
  endif()
endif()

if (NOT LibSci_LIBRARIES)
  find_package(BLAS REQUIRED)
  if (NOT WIN32)
      set(MATH_LIB "-lm")
  endif ()
  set(CMAKE_REQUIRED_LIBRARIES ${BLAS_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${MATH_LIB})
  include(CheckSymbolExists)
  check_symbol_exists(cblas_cgemm cblas.h LibSci_WORKS)
  unset(CMAKE_REQUIRED_INCLUDES)
  unset(CMAKE_REQUIRED_LIBRARIES)

  if (LibSci_WORKS)
    set(LibSci_LIBRARIES ${BLAS_LIBRARIES})
  endif()
endif()

if (LibSci_INCLUDE_DIRS AND LibSci_LIBRARIES)
  find_package_handle_standard_args(LibSci REQUIRED_VARS LibSci_INCLUDE_DIRS LibSci_LIBRARIES)

elseif (LibSci_INCLUDE_DIRS)
  find_package_handle_standard_args(LibSci REQUIRED_VARS LibSci_INCLUDE_DIRS)

elseif (LibSci_LIBRARIES)
  find_package_handle_standard_args(LibSci REQUIRED_VARS LibSci_LIBRARIES)

else()
  find_package_handle_standard_args(LibSci "Can't find LibSci")
endif()
