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
#include "core_lapack.h"

/***************************************************************************//**
 *
 * @ingroup core_symm
 *
 *  Performs one of the matrix-matrix operations
 *
 *     \f[ C = \alpha \times A \times B + \beta \times C \f]
 *  or
 *     \f[ C = \alpha \times B \times A + \beta \times C \f]
 *
 *  where alpha and beta are scalars, A is a symmetric matrix and B and
 *  C are m-by-n matrices.
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
 *          A is an lda-by-ka matrix, where ka is m when side = PlasmaLeft,
 *          and is n otherwise. Only the uplo triangular part is referenced.
 *
 * @param[in] lda
 *          The leading dimension of the array A. lda >= max(1,ka).
 *
 * @param[in] B
 *          B is an ldb-by-n matrix, where the leading m-by-n part of
 *          the array B must contain the matrix B.
 *
 * @param[in] ldb
 *          The leading dimension of the array B. ldb >= max(1,m).
 *
 * @param[in] beta
 *          The scalar beta.
 *
 * @param[in,out] C
 *          C is an ldc-by-n matrix.
 *          On exit, the array is overwritten by the m-by-n updated matrix.
 *
 * @param[in] ldc
 *          The leading dimension of the array C. ldc >= max(1,m).
 *
 ******************************************************************************/
__attribute__((weak))
void plasma_core_zsymm(plasma_enum_t side, plasma_enum_t uplo,
                int m, int n,
                plasma_complex64_t alpha, const plasma_complex64_t *A, int lda,
                                          const plasma_complex64_t *B, int ldb,
                plasma_complex64_t beta,        plasma_complex64_t *C, int ldc)
{
    cblas_zsymm(CblasColMajor,
                (CBLAS_SIDE)side, (CBLAS_UPLO)uplo,
                m, n,
                CBLAS_SADDR(alpha), A, lda,
                                    B, ldb,
                CBLAS_SADDR(beta),  C, ldc);
}

/******************************************************************************/
void plasma_core_omp_zsymm(
    plasma_enum_t side, plasma_enum_t uplo,
    int m, int n,
    plasma_complex64_t alpha, const plasma_complex64_t *A, int lda,
                              const plasma_complex64_t *B, int ldb,
    plasma_complex64_t beta,        plasma_complex64_t *C, int ldc,
    plasma_sequence_t *sequence, plasma_request_t *request)
{
    int ak;
    if (side == PlasmaLeft)
        ak = m;
    else
        ak = n;

    #pragma omp task depend(in:A[0:lda*ak]) \
                     depend(in:B[0:ldb*n]) \
                     depend(inout:C[0:ldc*n])
    {
        if (sequence->status == PlasmaSuccess)
            plasma_core_zsymm(side, uplo,
                       m, n,
                       alpha, A, lda,
                              B, ldb,
                       beta,  C, ldc);
    }
}
