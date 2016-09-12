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

/***************************************************************************//**
 *
 * @ingroup core_tradd
 *
 *  Performs an addition of two trapezoidal matrices similarly to the
 * 'pztradd()' function from the PBLAS library:
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
 * @param[in] uplo
 *          Specifies the shape of A and B matrices:
 *          - PlasmaFull: A and B are general matrices.
 *          - PlasmaUpper: op( A ) and B are upper trapezoidal matrices.
 *          - PlasmaLower: op( A ) and B are lower trapezoidal matrices.
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
 *          n >= 0.
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
 *          when transA = PlasmaNoTrans and n otherwise.
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
 *          ldb >= max(1,m).
 *
 ******************************************************************************/
void CORE_ztradd(PLASMA_enum uplo, PLASMA_enum transA, int m, int n,
                       PLASMA_Complex64_t  alpha,
                 const PLASMA_Complex64_t *A, int lda,
                       PLASMA_Complex64_t  beta,
                       PLASMA_Complex64_t *B, int ldb)
{
    int i, j;

    if (uplo == PlasmaFull) {
        CORE_zgeadd(transA, m, n, alpha, A, lda, beta, B, ldb);
        return;
    }

    if ((uplo != PlasmaUpper) &&
        (uplo != PlasmaLower)) {
        plasma_error("illegal value of uplo");
        return;
    }

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
        plasma_error("Illegal value of m");
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

    //=============
    // PlasmaLower
    //=============
    if (uplo == PlasmaLower) {
        switch( transA ) {
        case PlasmaConjTrans:
            for (j=0; j<n; j++, A++) {
                for(i=j; i<m; i++, B++) {
                    *B = beta * (*B) + alpha * conj(A[lda*i]);
                }
                B += ldb-m+j+1;
            }
            break;

        case PlasmaTrans:
            for (j=0; j<n; j++, A++) {
                for(i=j; i<m; i++, B++) {
                    *B = beta * (*B) + alpha * A[lda*i];
                }
                B += ldb-m+j+1;
            }
            break;

        case PlasmaNoTrans:
        default:
            for (j=0; j<n; j++) {
                for(i=j; i<m; i++, B++, A++) {
                    *B = beta * (*B) + alpha * (*A);
                }
                B += ldb-m+j+1;
                A += lda-m+j+1;
            }
        }
    }
    //=============
    // PlasmaUpper
    //=============
    else {
        switch( transA ) {
        case PlasmaConjTrans:
            for (j=0; j<n; j++, A++) {
                int mm = imin( j+1, m );
                for(i=0; i<mm; i++, B++) {
                    *B = beta * (*B) + alpha * conj(A[lda*i]);
                }
                B += ldb-mm;
            }
            break;

        case PlasmaTrans:
            for (j=0; j<n; j++, A++) {
                int mm = imin( j+1, m );
                for(i=0; i<mm; i++, B++) {
                    *B = beta * (*B) + alpha * (A[lda*i]);
                }
                B += ldb-mm;
            }
            break;

        case PlasmaNoTrans:
        default:
            for (j=0; j<n; j++) {
                int mm = imin( j+1, m );
                for(i=0; i<mm; i++, B++, A++) {
                    *B = beta * (*B) + alpha * (*A);
                }
                B += ldb-mm;
                A += lda-mm;
            }
        }
    }
}

/******************************************************************************/
void CORE_OMP_ztradd(
    PLASMA_enum uplo, PLASMA_enum transA, int m, int n,
    PLASMA_Complex64_t alpha, const PLASMA_Complex64_t *A, int lda,
    PLASMA_Complex64_t beta,        PLASMA_Complex64_t *B, int ldb)
{
    // omp depend assumes lda = PlasmaNoTrans ? m : n; ldb = m
    #pragma omp task depend(in:A[0:m*n]) depend(inout:B[0:m*n])
    CORE_ztradd(uplo, transA, m, n, alpha, A, lda, beta, B, ldb);
}
