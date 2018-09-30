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
 * @ingroup core_potrf
 *
 *  Performs the Cholesky factorization of a Hermitian positive definite
 *  matrix A. The factorization has the form
 *
 *    \f[ A = L \times L^H, \f]
 *    or
 *    \f[ A = U^H \times U, \f]
 *
 *  where U is an upper triangular matrix and L is a lower triangular matrix.
 *
 *******************************************************************************
 *
 * @param[in] uplo
 *          - PlasmaUpper: Upper triangle of A is stored;
 *          - PlasmaLower: Lower triangle of A is stored.
 *
 * @param[in] n
 *          The order of the matrix A. n >= 0.
 *
 * @param[in,out] A
 *          On entry, the Hermitian positive definite matrix A.
 *          If uplo = PlasmaUpper, the leading N-by-N upper triangular part of A
 *          contains the upper triangular part of the matrix A, and the strictly
 *          lower triangular part of A is not referenced.
 *          If uplo = PlasmaLower, the leading N-by-N lower triangular part of A
 *          contains the lower triangular part of the matrix A, and the strictly
 *          upper triangular part of A is not referenced.
 *          On exit, if return value = 0, the factor U or L from the Cholesky
 *          factorization A = U^H*U or A = L*L^H.
 *
 * @param[in] lda
 *          The leading dimension of the array A. lda >= max(1,n).
 *
 ******************************************************************************/
__attribute__((weak))
int plasma_core_zpotrf(plasma_enum_t uplo,
                 int n,
                 plasma_complex64_t *A, int lda)
{
    return LAPACKE_zpotrf_work(LAPACK_COL_MAJOR,
                               lapack_const(uplo),
                               n,
                               A, lda);
}

/******************************************************************************/
void plasma_core_omp_zpotrf(plasma_enum_t uplo,
                     int n,
                     plasma_complex64_t *A, int lda,
                     int iinfo,
                     plasma_sequence_t *sequence, plasma_request_t *request)
{
    #pragma omp task depend(inout:A[0:lda*n])
    {
        if (sequence->status == PlasmaSuccess) {
            int info = plasma_core_zpotrf(uplo,
                                   n,
                                   A, lda);
            if (info != 0)
                plasma_request_fail(sequence, request, iinfo+info);
        }
    }
}
