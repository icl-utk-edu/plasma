
#.rst:
# FindCBLAS
# --------
#
# Find CBLAS headers and library
# This module sets the following variables:
#
# ::
#
#   CBLAS_FOUND - set to true if a library implementing the BLAS interface
#     is found

find_path(PLASMA_MKL_CBLAS_PATH mkl_cblas.h "$ENV{MKLROOT}" "$ENV{MKLROOT}/include")
if ( PLASMA_MKL_CBLAS_PATH )
  message( STATUS "Found Intel MKL CBLAS header in ${PLASMA_MKL_CBLAS_PATH}")
  include_directories(${PLASMA_MKL_CBLAS_PATH})

  set( PLASMA_WITH_MKL TRUE )
endif ( PLASMA_MKL_CBLAS_PATH )

#include(CheckIncludeFiles)
#set(CBLAS_MKL_FOUND "FALSE")
#CHECK_INCLUDE_FILES( mkl_cblas.h;mkl_lapacke.h PLASMA_HAVE_MKL_CBLAS)
#message( STATUS "Result ${CBLAS_MKL_FOUND}")
