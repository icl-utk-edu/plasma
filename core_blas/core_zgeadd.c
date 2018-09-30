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
#include "plasma_internal.h"
#include "plasma_types.h"
#include "core_lapack.h"

/****************************************************************************//*
 *
 * @ingroup core_geadd
 *
 *  Performs an addition of two general matrices similarly to the
 *  pzgeadd() function from the PBLAS library:
 *
 *    \f[ B = \alpha * op( A ) + \beta * B, \f]
 *
 *  where op( X ) is one of:
 *    \f[ op( X ) = X,   \f]
 *    \f[ op( X ) = X^T, \f]
 *    \f[ op( X ) = X^H, \f]
 *
 *  alpha and beta are scalars and A, B are matrices with op( A ) an m-by-n or
 *  n-by-m matrix depending on the value of transa and B an m-by-n matrix.
 *
 *******************************************************************************
 *
 * @param[in] transa
 *          Specifies whether the matrix A is non-transposed, transposed, or
 *          conjugate transposed
 *          - PlasmaNoTrans:   op( A ) = A
 *          - PlasmaTrans:     op( A ) = A^T
 *          - PlasmaConjTrans: op( A ) = A^H
 *
 * @param[in] m
 *          Number of rows of the matrices op( A ) and B.
 *          m >= 0.
 *
 * @param[in] n
 *          Number of columns of the matrices op( A ) and B.
 *
 * @param[in] alpha
 *          Scalar factor of A.
 *
 * @param[in] A
 *          Matrix of size lda-by-k, where k is n when transa == PlasmaNoTrans
 *          and m otherwise.
 *
 * @param[in] lda
 *          Leading dimension of the array A. lda >= max(1,l), where l is m
 *          when transa == PlasmaNoTrans and n otherwise.
 *
 * @param[in] beta
 *          Scalar factor of B.
 *
 * @param[in,out] B
 *          Matrix of size ldb-by-n.
 *          On exit, B = alpha * op( A ) + beta * B
 *
 * @param[in] ldb
 *          Leading dimension of the array B.
 *          ldb >= max(1,m)
 *
 ******************************************************************************/
__attribute__((weak))
int plasma_core_zgeadd(plasma_enum_t transa,
                int m, int n,
                plasma_complex64_t alpha, const plasma_complex64_t *A, int lda,
                plasma_complex64_t beta,        plasma_complex64_t *B, int ldb)
{
    // Check input arguments.
    if ((transa != PlasmaNoTrans) &&
        (transa != PlasmaTrans)   &&
        (transa != PlasmaConjTrans)) {
        plasma_coreblas_error("illegal value of transa");
        return -1;
    }
    if (m < 0) {
        plasma_coreblas_error("illegal value of m");
        return -2;
    }
    if (n < 0) {
        plasma_coreblas_error("illegal value of n");
        return -3;
    }
    if (A == NULL) {
        plasma_coreblas_error("NULL A");
        return -5;
    }
    if ((transa == PlasmaNoTrans && lda < imax(1, m) && (m > 0)) ||
        (transa != PlasmaNoTrans && lda < imax(1, n) && (n > 0))) {
        plasma_coreblas_error("illegal value of lda");
        return -6;
    }
    if (B == NULL) {
        plasma_coreblas_error("NULL B");
        return -8;
    }
    if ((ldb < imax(1, m)) && (m > 0)) {
        plasma_coreblas_error("illegal value of ldb");
        return -9;
    }

    // quick return
    if (m == 0 || n == 0 || (alpha == 0.0 && beta == 1.0))
        return PlasmaSuccess;

    switch (transa) {
    case PlasmaConjTrans:
        for (int j = 0; j < n; j++)
            for (int i = 0; i < m; i++)
                B[ldb*j+i] = beta * B[ldb*j+i] + alpha * conj(A[lda*i+j]);
        break;

    case PlasmaTrans:
        for (int j = 0; j < n; j++)
            for (int i = 0; i < m; i++)
                B[ldb*j+i] = beta * B[ldb*j+i] + alpha * A[lda*i+j];
        break;

    case PlasmaNoTrans:
        for (int j = 0; j < n; j++)
            for (int i = 0; i < m; i++)
                B[ldb*j+i] = beta * B[ldb*j+i] + alpha * A[lda*j+i];
    }

    return PlasmaSuccess;
}

/******************************************************************************/
void plasma_core_omp_zgeadd(
    plasma_enum_t transa,
    int m, int n,
    plasma_complex64_t alpha, const plasma_complex64_t *A, int lda,
    plasma_complex64_t beta,        plasma_complex64_t *B, int ldb,
    plasma_sequence_t *sequence, plasma_request_t *request)
{
    int k = (transa == PlasmaNoTrans) ? n : m;

    #pragma omp task depend(in:A[0:lda*k]) \
                     depend(inout:B[0:ldb*n])
    {
        if (sequence->status == PlasmaSuccess) {
            int retval = plasma_core_zgeadd(transa,
                                     m, n,
                                     alpha, A, lda,
                                     beta,  B, ldb);
            if (retval != PlasmaSuccess) {
                plasma_error("core_zgeadd() failed");
                plasma_request_fail(sequence, request, PlasmaErrorInternal);
            }
        }
    }
}
