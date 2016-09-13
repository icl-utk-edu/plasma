/**
 *
 * @file
 *
 *  PLASMA is a software package provided by:
 *  University of Tennessee, US,
 *  University of Manchester, UK.
 *
 * @precisions normal z -> c d s
 *
 **/

#include "core_blas.h"
#include "plasma_types.h"
#include "core_lapack.h"

/***************************************************************************//**
 *
 * @ingroup core_lacpy
 *
 *  Copies all or part of a two-dimensional matrix A to another matrix B.
 *
 *******************************************************************************
 *
 * @param[in] uplo
 *          - PlasmaFull:  entire A,
 *          - PlasmaUpper: upper triangle,
 *          - PlasmaLower: lower triangle.
 *
 * @param[in] m
 *          The number of rows of the matrices A and B.
 *          m >= 0.
 *
 * @param[in] n
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
    LAPACKE_zlacpy_work(LAPACK_COL_MAJOR,
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
    // omp depends assume lda == ldb == m.
    #pragma omp task depend(in:A[0:m*n]) depend(out:B[0:m*n])
    CORE_zlacpy(uplo,
                m, n,
                A, lda,
                B, ldb);
}
