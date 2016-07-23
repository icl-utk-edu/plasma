/**
 *
 * @file core_zsyr2k.c
 *
 *  PLASMA core_blas kernel.
 *  PLASMA is a software package provided by Univ. of Tennessee,
 *  Univ. of California Berkeley, Univ. of Colorado Denver and
 *  Univ. of Manchester.
 *
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
 * @ingroup core_syr2k
 *
 *  CORE_zsyr2k - Performs one of the symmetric rank 2k operations
 *
 *    \f[ C = \alpha A \times B^T + \alpha B \times A^T + \beta C \f],
 *    or
 *    \f[ C = \alpha A^T \times B + \alpha B^T \times A + \beta C \f],
 *
 *  where alpha and beta are complex scalars, C is an n-by-n symmetric
 *  matrix, and A and B are n-by-k matrices the first case and k-by-n
 *  matrices in the second case.
 *
 *******************************************************************************
 *
 * @param[in] uplo
 *          - PlasmaUpper: Upper triangle of C is stored;
 *          - PlasmaLower: Lower triangle of C is stored.
 *
 * @param[in] trans
 *          - PlasmaNoTrans:
 *            \f[ C = \alpha A \times B^T + \alpha B \times A^T + \beta C \f];
 *          - PlasmaTrans:
 *            \f[ C = \alpha A^T \times B + \alpha B^T \times A + \beta C \f].
 *
 * @param[in] n
 *          The order of the matrix C. n must be at least zero.
 *
 * @param[in] k
 *          k specifies the number of columns of the A and
 *          B matrices with trans = PlasmaNoTrans, or specifies
 *          the number of rows of the A and B matrices with trans = PlasmaTrans.
 *
 * @param[in] alpha
 *          The scalar alpha.
 *
 * @param[in] A
 *          A lda-by-ka matrix, where ka is k when trans = PlasmaNoTrans,
 *          and is n otherwise.
 *
 * @param[in] lda
 *          The leading dimension of the array A. lda must be at least
 *          max( 1, n ), otherwise lda must be at least max( 1, k ).
 *
 * @param[in] B
 *          A ldb-by-kb matrix, where kb is k when trans = PlasmaNoTrans,
 *          and is n otherwise.
 *
 * @param[in] ldb
 *          The leading dimension of the array B. ldb must be at least
 *          max( 1, n ), otherwise ldb must be at least max( 1, k ).
 *
 * @param[in] beta
 *          The scalar beta.
 *
 * @param[in,out] C
 *          C  ldc-by-n matrix.
 *          On exit, the array uplo part of the matrix is overwritten
 *          by the uplo part of the updated matrix.
 *
 * @param[in] ldc
 *          The leading dimension of the array C. ldc >= max( 1, n ).
 *
 ******************************************************************************/
void CORE_zsyr2k(PLASMA_enum uplo, PLASMA_enum trans,
                 int n, int k,
                 PLASMA_Complex64_t alpha, const PLASMA_Complex64_t *A, int lda,
                                           const PLASMA_Complex64_t *B, int ldb,
                 PLASMA_Complex64_t beta,        PLASMA_Complex64_t *C, int ldc)
{
    cblas_zsyr2k(CblasColMajor,
                 (CBLAS_UPLO)uplo, (CBLAS_TRANSPOSE)trans,
                 n, k,
                 CBLAS_SADDR(alpha), A, lda,
                                     B, ldb,
                 CBLAS_SADDR(beta),  C, ldc);
}

/******************************************************************************/
void CORE_OMP_zsyr2k(
    PLASMA_enum uplo, PLASMA_enum trans,
    int n, int k,
    PLASMA_Complex64_t alpha, const PLASMA_Complex64_t *A, int lda,
                              const PLASMA_Complex64_t *B, int ldb,
    PLASMA_Complex64_t beta,        PLASMA_Complex64_t *C, int ldc)
{
    // omp depends assume lda == n or k, ldb == n or k, and ldc == n,
    // depending on transposes
    #pragma omp task depend(in:A[0:n*k]) \
                     depend(in:B[0:n*k]) \
                     depend(inout:C[0:n*n])
    CORE_zsyr2k(uplo, trans,
                n, k,
                alpha, A, lda,
                       B, ldb,
                beta,  C, ldc);
}
