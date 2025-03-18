/**
 *
 * @file
 *
 *  PLASMA is a software package provided by:
 *  University of Tennessee, US,
 *  University of Manchester, UK.
 *
 * @precisions normal z -> c
 *
 **/

#include <plasma_core_blas.h>
#include "plasma_types.h"
#include "core_lapack.h"

/***************************************************************************//**
 *
 * @ingroup core_hemm
 *
 *  Performs one of the matrix-matrix operations
 *  \[
 *      C = \alpha A B + \beta C,
 *  \]
 *  or
 *  \[
 *      C = \alpha B A + \beta C,
 *  \]
 *  where alpha and beta are scalars, A is a Hermitian matrix and B and
 *  C are m-by-n matrices.
 *
 *******************************************************************************
 *
 * @param[in] side
 *          Specifies whether the Hermitian matrix A appears on the
 *          left or right in the operation as follows:
 *          - PlasmaLeft:  \[ C = \alpha A B + \beta C \]
 *          - PlasmaRight: \[ C = \alpha B A + \beta C \]
 *
 * @param[in] uplo
 *          Specifies whether the upper or lower triangular part of
 *          the Hermitian matrix A is to be referenced as follows:
 *          - PlasmaLower:     Only the lower triangular part of the
 *                             Hermitian matrix A is to be referenced.
 *          - PlasmaUpper:     Only the upper triangular part of the
 *                             Hermitian matrix A is to be referenced.
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
void plasma_core_zhemm(plasma_enum_t side, plasma_enum_t uplo,
                int m, int n,
                plasma_complex64_t alpha, const plasma_complex64_t *A, int lda,
                                          const plasma_complex64_t *B, int ldb,
                plasma_complex64_t beta,        plasma_complex64_t *C, int ldc)
{
    cblas_zhemm(CblasColMajor,
                (CBLAS_SIDE)side, (CBLAS_UPLO)uplo,
                m, n,
                CBLAS_SADDR(alpha), A, lda,
                                    B, ldb,
                CBLAS_SADDR(beta),  C, ldc);
}

/******************************************************************************/
void plasma_core_omp_zhemm(
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
            plasma_core_zhemm(side, uplo,
                       m, n,
                       alpha, A, lda,
                              B, ldb,
                       beta,  C, ldc);
    }
}
