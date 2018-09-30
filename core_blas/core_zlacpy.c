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

#include <plasma_core_blas.h>
#include "plasma_types.h"
#include "plasma_internal.h"
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
 *          - PlasmaGeneral: entire A,
 *          - PlasmaUpper:   upper triangle,
 *          - PlasmaLower:   lower triangle.
 *
 * @param[in] transa
 *          - PlasmaNoTrans:   A is not transposed,
 *          - PlasmaTrans:     A is transposed,
 *          - PlasmaConjTrans: A is conjugate transposed.
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
 *          ldb >= max(1,m).
 *
 ******************************************************************************/
__attribute__((weak))
void plasma_core_zlacpy(plasma_enum_t uplo, plasma_enum_t transa,
                 int m, int n,
                 const plasma_complex64_t *A, int lda,
                       plasma_complex64_t *B, int ldb)
{
    if (transa == PlasmaNoTrans) {
        LAPACKE_zlacpy_work(LAPACK_COL_MAJOR,
                            lapack_const(uplo),
                            m, n,
                            A, lda,
                            B, ldb);
    }
    else if (transa == PlasmaTrans) {
        switch (uplo) {
        case PlasmaUpper:
            for (int i = 0; i < imin(m, n); i++)
                for (int j = i; j < n; j++)
                    B[j + i*ldb] = A[i + j*lda];
            break;
        case PlasmaLower:
            for (int i = 0; i < m; i++)
                for (int j = 0; j <= imin(i, n); j++)
                    B[j + i*ldb] = A[i + j*lda];
            break;
        case PlasmaGeneral:
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                    B[j + i*ldb] = A[i + j*lda];
            break;
        }
    }
    else {
        switch (uplo) {
        case PlasmaUpper:
            for (int i = 0; i < imin(m, n); i++)
                for (int j = i; j < n; j++)
                    B[j + i*ldb] = conj(A[i + j*lda]);
            break;
        case PlasmaLower:
            for (int i = 0; i < m; i++)
                for (int j = 0; j <= imin(i, n); j++)
                    B[j + i*ldb] = conj(A[i + j*lda]);
            break;
        case PlasmaGeneral:
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                    B[j + i*ldb] = conj(A[i + j*lda]);
            break;
        }
    }
}

/******************************************************************************/
void plasma_core_omp_zlacpy(plasma_enum_t uplo, plasma_enum_t transa,
                     int m, int n,
                     const plasma_complex64_t *A, int lda,
                           plasma_complex64_t *B, int ldb,
                     plasma_sequence_t *sequence, plasma_request_t *request)
{
    #pragma omp task depend(in:A[0:lda*n]) \
                     depend(out:B[0:ldb*n])
    {
        if (sequence->status == PlasmaSuccess)
            plasma_core_zlacpy(uplo, transa,
                        m, n,
                        A, lda,
                        B, ldb);
    }
}
