/**
 *
 * @file core_ztrsm.c
 *
 *  PLASMA core_blas kernel.
 *  PLASMA is a software package provided by Univ. of Tennessee,
 *  Univ. of California Berkeley and Univ. of Colorado Denver.
 * @version 3.0.0
 * @author Mawussi Zounon
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
 * @ingroup core_trsm
 *
 *  Performs one of the matrix equations
 *
 *    \f[ op( A )\times X  = \alpha B, \f] or
 *    \f[ op( X )\times A  = \alpha B, \f]
 *
 *  where op( X ) is one of:
 *          - op( X ) = X   or
 *          - op( X ) = X^T or
 *          - op( X ) = X^H,
 *
 *  alpha is a scalar, A and  are B m by n  matrices.
 *  The matrix X is overwritten on B
 *
 *******************************************************************************
 *
 * @param[in] side
 *          - PlasmaLeft:  A*X = B
 *          - PlasmaRight: X*A = B
 *
 * @param[in] uplo
 *          Specifies whether A is upper triangular or lower triangular:
 *          - PlasmaUpper: Upper triangle of A is stored,
 *          - PlasmaLower: Lower triangle of A is stored.
 *
 * @param[in] transA
 *          - PlasmaNoTrans:   A is transposed;
 *          - PlasmaTrans:     A is not transposed;
 *          - PlasmaConjTrans: A is conjugate transposed.
 *
 * @param[in] diag
 *          - PlasmaNonUnit: A is non unit;
 *          - PlasmaUnit:    A us unit.
 *
 * @param[in] m
 *          The order of the matrix A. M >= 0.
 *
 * @param[in] n
 *          The number of columns of the matrix B. N >= 0.
 *
 * @param[in] alpha
 *          The scalar alpha.
 *
 * @param[in] A
 *          The triangular matrix. If uplo = PlasmaUpper, the leading m-by-m
 *          upper triangular part of the array A contains the upper triangular
 *          matrix, and the strictly lower
 *          triangular part of A is not referenced. If uplo = PlasmaLower,
 *          the leading m-by-m  lower triangular part of the array A contains
 *          the lower triangular matrix, and the strictly upper triangular part
 *          of A is not referenced. If diag = PlasmaUnit, the diagonal elements
 *          of A are also not referenced and are assumed to be 1.
 *
 * @param[in] lda
 *          The leading dimension of the array A. lda >= max(1,m).
 *
 * @param[in,out] B
 *          On entry, the m-by-n right hand side matrix B.
 *          On exit, if return value = 0, the m-by-n solution matrix X.
 *
 * @param[in] ldb
 *          The leading dimension of the array B. LDB >= max(1,m).
 *
 ******************************************************************************/
void CORE_ztrsm(PLASMA_enum side, PLASMA_enum uplo,
                PLASMA_enum transA, PLASMA_enum diag,
                int m, int n,
                PLASMA_Complex64_t alpha, const PLASMA_Complex64_t *A, int lda,
                                                PLASMA_Complex64_t *B, int ldb)
{
    cblas_ztrsm(CblasColMajor,
                (CBLAS_SIDE)side, (CBLAS_UPLO)uplo,
                (CBLAS_TRANSPOSE)transA, (CBLAS_DIAG)diag,
                m, n,
                CBLAS_SADDR(alpha), A, lda,
                                    B, ldb);
}

/******************************************************************************/
void CORE_OMP_ztrsm(
    PLASMA_enum side, PLASMA_enum uplo,
    PLASMA_enum transA, PLASMA_enum diag,
    int m, int n,
    PLASMA_Complex64_t alpha, const PLASMA_Complex64_t *A, int lda,
                                    PLASMA_Complex64_t *B, int ldb)
{
    // omp depends assume lda == m or n, ldb == m
    // depending on side
    #pragma omp task depend(in:A[0:m*m]) depend(inout:B[0:m*n])
    CORE_ztrsm(side, uplo,
               transA, diag,
               m, n,
               alpha, A, lda,
                      B, ldb);
}
