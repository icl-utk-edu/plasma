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
 * @ingroup core_trtri
 *
 *  Computes the inverse of an upper or lower
 *  triangular matrix A.
 *
 *******************************************************************************
 *
 * @param[in] uplo
 *          = PlasmaUpper: Upper triangle of A is stored;
 *          = PlasmaLower: Lower triangle of A is stored.
 *
 * @param[in] diag
 *          = PlasmaNonUnit: A is non-unit triangular;
 *          = PlasmaUnit:    A is unit triangular.
 *
 * @param[in] n
 *          The order of the matrix A. n >= 0.
 *
 * @param[in,out] A
 *          On entry, the triangular matrix A.  If uplo = 'U', the
 *          leading n-by-n upper triangular part of the array A
 *          contains the upper triangular matrix, and the strictly
 *          lower triangular part of A is not referenced.  If uplo =
 *          'L', the leading n-by-n lower triangular part of the array
 *          A contains the lower triangular matrix, and the strictly
 *          upper triangular part of A is not referenced.  If diag =
 *          'U', the diagonal elements of A are also not referenced and
 *          are assumed to be 1.  On exit, the (triangular) inverse of
 *          the original matrix.
 *
 * @param[in] lda
 *          The leading dimension of the array A. lda >= max(1,n).
 *
 * @retval PlasmaSuccess on successful exit
 * @retval < 0 if -i, the i-th argument had an illegal value
 * @retval > 0 if i, A(i,i) is exactly zero.  The triangular
 *          matrix is singular and its inverse can not be computed.
 *
 ******************************************************************************/
__attribute__((weak))
int plasma_core_ztrtri(plasma_enum_t uplo, plasma_enum_t diag,
                 int n,
                 plasma_complex64_t *A, int lda)
{
    return LAPACKE_ztrtri_work(LAPACK_COL_MAJOR,
                        lapack_const(uplo), lapack_const(diag),
                        n, A, lda);
}

/******************************************************************************/
void plasma_core_omp_ztrtri(plasma_enum_t uplo, plasma_enum_t diag,
                     int n,
                     plasma_complex64_t *A, int lda,
                     int iinfo,
                     plasma_sequence_t *sequence, plasma_request_t *request)
{
    #pragma omp task depend(inout:A[0:lda*n])
    {
        if (sequence->status == PlasmaSuccess) {
            int info = plasma_core_ztrtri(uplo, diag,
                                   n, A, lda);
            if (info != 0)
                plasma_request_fail(sequence, request, iinfo+info);
        }
    }
}
