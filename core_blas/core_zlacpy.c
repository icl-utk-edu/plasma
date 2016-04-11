/**
 *
 * @file core_zgemm.c
 *
 *  PLASMA core_blas kernel.
 *  PLASMA is a software package provided by Univ. of Tennessee,
 *  Univ. of California Berkeley and Univ. of Colorado Denver.
 *
 * @version 3.0.0
 * @author Jakub Kurzak
 * @date 2016-01-01
 * @precisions normal z -> c d s
 *
 **/

#include "core_blas.h"
#include "plasma_types.h"

#ifdef PLASMA_WITH_MKL
    #include <mkl.h>
#else
    #include <cblas.h>
    #include <lapacke.h>
#endif

/***************************************************************************//**
 *
 * @ingroup CORE_PLASMA_Complex64_t
 *
 *  Copies all or part of a two-dimensional matrix A to another matrix B.
 *
 *******************************************************************************
 *
 * @param[in] uplo
 *          - PlasmaUpperLower: entire A,
 *          - PlasmaUpper:      upper triangle,
 *          - PlasmaLower:      lower triangle.
 *
 * @param[in] M
 *          The number of rows of the matrices A and B.
 *          m >= 0.
 *
 * @param[in] N
 *          The number of columns of the matrices A and B.
 *          n >= 0.
 *
 * @param[in] A
 *          The m-by-n matrix to copy.
 *
 * @param[in] lda
 *          The leading dimension of the array A.
 *          lda >= max(1,m).
 *
 * @param[out] B
 *          The m-by-n copy of the matrix A.
 *          On exit, B = A ONLY in the locations specified by uplo.
 *
 * @param[in] ldb
 *          The leading dimension of the array B.
 *          ldb >= max(1,M).
 *
 ******************************************************************************/
void CORE_zlacpy(PLASMA_enum uplo,
                 int m, int n,
                 const PLASMA_Complex64_t *A, int lda,
                       PLASMA_Complex64_t *B, int ldb)
{
    LAPACKE_zlacpy(LAPACK_COL_MAJOR,
                   lapack_const(uplo),
                   m, n,
                   A, lda,
                   B, ldb);
}

/******************************************************************************/
void CORE_OMP_zlacpy(PLASMA_enum uplo,
                     int m, int n, int nb,
                     const PLASMA_Complex64_t *A, int lda,
                           PLASMA_Complex64_t *B, int ldb)
{
#pragma omp task depend(in:A[0:m*n]) depend(out:B[0:m*n]) 
    CORE_zlacpy(uplo,
                m, n,
                A, lda,
                B, ldb);
}
