
#.rst:
# FindCBLAS
# --------
#
# Find CBLAS header files and libraries
# This module sets the following variables:
#
# ::
#
#   CBLAS_FOUND - set to true if a library implementing the BLAS C interface
#     is found

find_path(PLASMA_MKL_CBLAS_PATH mkl_cblas.h "$ENV{MKLROOT}" "$ENV{MKLROOT}/include")
if ( PLASMA_MKL_CBLAS_PATH )
  message( STATUS "Found Intel MKL CBLAS header in ${PLASMA_MKL_CBLAS_PATH}")
  include_directories(${PLASMA_MKL_CBLAS_PATH})

  add_definitions( -DPLASMA_WITH_MKL ) # this adds command line option only
  set( PLASMA_WITH_MKL TRUE )

  set(CBLAS_FOUND TRUE)
endif ( PLASMA_MKL_CBLAS_PATH )

#include(CheckIncludeFiles)
#set(CBLAS_MKL_FOUND "FALSE")
#CHECK_INCLUDE_FILES( mkl_cblas.h;mkl_lapacke.h PLASMA_WITH_MKL_CBLAS)
#message( STATUS "Result ${CBLAS_MKL_FOUND}")
