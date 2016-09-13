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
#include "plasma_internal.h"
#include "plasma_types.h"
#include "core_lapack.h"

/****************************************************************************//*
 *
 * @ingroup core_geadd
 *
 *  Performs an addition of two general matrices similarly to the
 * 'pzgeadd()' function from the PBLAS library:
 *
 *    \f[ B = \alpha * op( A ) + \beta * B, \f]
 *
 *  where op( X ) is one of:
 *    \f[ op( X ) = X,   \f]
 *    \f[ op( X ) = X^T, \f]
 *    \f[ op( X ) = X^H, \f]
 *
 *  alpha and beta are scalars and A, B are matrices with op( A ) an m-by-n or
 *  n-by-m matrix depending on the value of transA and B an m-by-n matrix.
 *
 *******************************************************************************
 *
 * @param[in] transA
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
 *          Matrix of size lda-by-k, where k is n when transA == PlasmaNoTrans
 *          and m otherwise.
 *
 * @param[in] lda
 *          Leading dimension of the array A. lda >= max(1,l), where l is m
 *          when transA == PlasmaNoTrans and n otherwise.
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
void core_zgeadd(PLASMA_enum transA, int m, int n,
                      plasma_complex64_t  alpha,
                const plasma_complex64_t *A, int lda,
                      plasma_complex64_t  beta,
                      plasma_complex64_t *B, int ldb)
{
    int i, j;

    if ((transA != PlasmaNoTrans) &&
        (transA != PlasmaTrans)   &&
        (transA != PlasmaConjTrans)) {

        plasma_error("illegal value of transA");
        return;
    }

    if (m < 0) {
        plasma_error("Illegal value of m");
        return;
    }

    if (n < 0) {
        plasma_error("Illegal value of n");
        return;
    }

    if (A == NULL) {
        plasma_error("NULL A");
        return;
    }

    if ( ((transA == PlasmaNoTrans) && (lda < imax(1,m)) && (m > 0)) ||
         ((transA != PlasmaNoTrans) && (lda < imax(1,n)) && (n > 0)) ) {

        plasma_error("Illegal value of lda");
        return;
    }

    if (B == NULL) {
        plasma_error("NULL B");
        return;
    }

    if ( (ldb < imax(1,m)) && (m > 0) ) {
        plasma_error("Illegal value of ldb");
        return;
    }

    switch (transA) {
    case PlasmaConjTrans:
        for (j = 0; j < n; j++, A++) {
            for (i = 0; i < m; i++, B++) {
                *B = beta * (*B) + alpha * conj(A[lda*i]);
            }
            B += ldb-m;
        }
        break;

    case PlasmaTrans:
        for (j = 0; j < n; j++, A++) {
            for (i = 0; i < m; i++, B++) {
                *B = beta * (*B) + alpha * A[lda*i];
            }
            B += ldb-m;
        }
        break;

    case PlasmaNoTrans:
    default:
        for (j = 0; j < n; j++) {
            for (i = 0; i < m; i++, B++, A++) {
                *B = beta * (*B) + alpha * (*A);
            }
            A += lda-m;
            B += ldb-m;
        }
    }
}

/******************************************************************************/
void core_omp_zgeadd(
    PLASMA_enum transA, int m, int n,
    plasma_complex64_t alpha, const plasma_complex64_t *A, int lda,
    plasma_complex64_t beta,        plasma_complex64_t *B, int ldb)
{
    // omp depend assumes lda = PlasmaNoTrans ? m : n; ldb = m
    #pragma omp task depend(in:A[0:m*n]) depend(inout:B[0:m*n])
    core_zgeadd(transA, m, n, alpha, A, lda, beta, B, ldb);
}
