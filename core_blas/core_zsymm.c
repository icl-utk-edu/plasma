/**
 *
 * @file core_zsymm.c
 *
 *  PLASMA core_blas kernel.
 *  PLASMA is a software package provided by Univ. of Tennessee,
 *  Univ. of Manchester, Univ. of California Berkeley and
 *  Univ. of Colorado Denver
 *
 * @version 3.0.0
 * @author Samuel D. Relton
 * @date 2016-05-16
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
 * @ingroup core_hemm
 *
 *  Performs one of the matrix-matrix operations
 *
 *     \f[ C = \alpha \times A \times B + \beta \times C \f]
 *  or
 *     \f[ C = \alpha \times B \times A + \beta \times C \f]
 *
 *  where alpha and beta are scalars, A is a symmetric matrix and B and
 *  C are m by n matrices.
 *
 *******************************************************************************
 *
 * @param[in] side
 *          Specifies whether the symmetric matrix A appears on the
 *          left or right in the operation as follows:
 *          - PlasmaLeft:  \f[ C = \alpha \times A \times B + \beta \times C \f]
 *          - PlasmaRight: \f[ C = \alpha \times B \times A + \beta \times C \f]
 *
 * @param[in] uplo
 *          Specifies whether the upper or lower triangular part of
 *          the symmetric matrix A is to be referenced as follows:
 *          - PlasmaLower:     Only the lower triangular part of the
 *                             symmetric matrix A is to be referenced.
 *          - PlasmaUpper:     Only the upper triangular part of the
 *                             symmetric matrix A is to be referenced.
 *
 * @param[in] m
 *          The number of rows of the matrix C. m >= 0.
 *
 * @param[in] n
 *          The number of columns of the matrix C. n >= 0.
 *
 * @param[in] alpha
 *          The scalar alpha.
 *
 * @param[in] A
 *          A is a lda-by-ka matrix, where ka is m when side = PlasmaLeft,
 *          and is n otherwise. Only the uplo triangular part is referenced.
 *
 * @param[in] lda
 *          The leading dimension of the array A. lda >= max(1,ka).
 *
 * @param[in] B
 *          B is a ldb-by-n matrix, where the leading m-by-n part of
 *          the array B must contain the matrix B.
 *
 * @param[in] ldb
 *          The leading dimension of the array B. ldb >= max(1,m).
 *
 * @param[in] beta
 *          The scalar beta.
 *
 * @param[in,out] C
 *          C is a ldc-by-n matrix.
 *          On exit, the array is overwritten by the m-by-n updated matrix.
 *
 * @param[in] ldc
 *          The leading dimension of the array C. ldc >= max(1,m).
 *
 ******************************************************************************/
void CORE_zsymm(PLASMA_enum side, PLASMA_enum uplo,
                int m, int n,
                PLASMA_Complex64_t alpha, const PLASMA_Complex64_t *A, int lda,
                                          const PLASMA_Complex64_t *B, int ldb,
                PLASMA_Complex64_t beta,        PLASMA_Complex64_t *C, int ldc)
{
    cblas_zsymm(CblasColMajor,
                (CBLAS_SIDE)side, (CBLAS_UPLO)uplo,
                m, n,
                CBLAS_SADDR(alpha), A, lda,
                                    B, ldb,
                CBLAS_SADDR(beta),  C, ldc);
}

/******************************************************************************/
void CORE_OMP_zsymm(
    PLASMA_enum side, PLASMA_enum uplo,
    int m, int n,
    PLASMA_Complex64_t alpha, const PLASMA_Complex64_t *A, int lda,
                              const PLASMA_Complex64_t *B, int ldb,
    PLASMA_Complex64_t beta,        PLASMA_Complex64_t *C, int ldc)
{
    if (side == PlasmaLeft) {
        #pragma omp task depend(in:A[0:m*m]) \
                         depend(in:B[0:m*n]) \
                         depend(inout:C[0:m*n])
        CORE_zsymm(side, uplo,
                   m, n,
                   alpha, A, lda,
                          B, ldb,
                   beta,  C, ldc);
    }
    else {
        #pragma omp task depend(in:A[0:n*n]) \
                         depend(in:B[0:m*n]) \
                         depend(inout:C[0:m*n])
        CORE_zsymm(side, uplo,
                   m, n,
                   alpha, A, lda,
                          B, ldb,
                   beta,  C, ldc);
    }
}
