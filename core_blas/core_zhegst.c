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
 * @ingroup core_hegst
 *
 *  Reduces a complex Hermitian-definite generalized eigenproblem to standard
 *  form.
 *
 *  If ITYPE = 1, the problem is A*x = lambda*B*x,
 *  and A is overwritten by inv(U^H)*A*inv(U) or inv(L)*A*inv(L^H)
 *
 *  If ITYPE = 2 or 3, the problem is A*B*x = lambda*x or
 *  B*A*x = lambda*x, and A is overwritten by U*A*U^H or L^H*A*L.
 *
 *******************************************************************************
 *
 * @param[in] itype
 *          = 1: compute inv(U^H)*A*inv(U) or inv(L)*A*inv(L^H);
 *          = 2 or 3: compute U*A*U^H or L^H*A*L.
 *
 * @param[in] uplo
 *          If PlasmaUpper, upper triangle of A is stored and B is factored as
 *          U^H*U;
 *          If PlasmaLower, lower triangle of A is stored and B is factored as
 *          L*L^H.
 *
 * @param[in] n
 *          The order of the matrices A and B.  N >= 0.
 *
 * @param[in,out] A
 *          On entry, the Hermitian matrix A.  If UPLO = 'U', the leading
 *          N-by-N upper triangular part of A contains the upper
 *          triangular part of the matrix A, and the strictly lower
 *          triangular part of A is not referenced.  If UPLO = 'L', the
 *          leading N-by-N lower triangular part of A contains the lower
 *          triangular part of the matrix A, and the strictly upper
 *          triangular part of A is not referenced.
 *
 *          On exit, if INFO = 0, the transformed matrix, stored in the
 *          same format as A.
 *
 * @param[in] lda
 *          The leading dimension of the array A.  LDA >= max(1,N).
 *
 * @param[in,out] B
 *          The triangular factor from the Cholesky factorization of B,
 *          as returned by ZPOTRF.
 *
 * @param[in] ldb
 *          The leading dimension of the array B.  LDB >= max(1,N).
 *
 ******************************************************************************/
__attribute__((weak))
int plasma_core_zhegst(int itype, plasma_enum_t uplo,
                int n,
                plasma_complex64_t *A, int lda,
                plasma_complex64_t *B, int ldb)
{
    int info = LAPACKE_zhegst_work(
        LAPACK_COL_MAJOR,
        itype,
        lapack_const(uplo),
        n, A, lda, B, ldb );
    return info;
}

/******************************************************************************/
void plasma_core_omp_zhegst(int itype, plasma_enum_t uplo,
                     int n,
                     plasma_complex64_t *A, int lda,
                     plasma_complex64_t *B, int ldb,
                     plasma_sequence_t *sequence,
                     plasma_request_t *request)
{
    #pragma omp task depend(inout:A[0:lda*n]) \
                     depend(in:B[0:ldb*n])
    {
        if (sequence->status == PlasmaSuccess)
            plasma_core_zhegst(itype, uplo,
                        n,
                        A, lda,
                        B, ldb);
    }
}
