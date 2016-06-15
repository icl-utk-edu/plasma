/**
 *
 * @file core_zpotrf.c
 *
 *  PLASMA core_blas kernel.
 *  PLASMA is a software package provided by Univ. of Tennessee,
 *  Univ. of California Berkeley and Univ. of Colorado Denver and
 *  Univ. of Manchester.
 *
 * @version 
 * @author Pedro V. Lara
 * @date 
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
 * @ingroup PLASMA_Complex64_t
 *
 *  Performs the Cholesky factorization of a symmetric positive definite
 *  (or Hermitian positive definite in the complex case) matrix A.
 *  The factorization has the form
 *
 *    \f[ A = L \times L^H \f],
 *    or
 *    \f[ A = U^H \times U \f],
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
 *          On entry, the symmetric positive definite (or Hermitian) matrix A.
 *          If uplo = PlasmaUpper, the leading N-by-N upper triangular part of A
 *          contains the upper triangular part of the matrix A, and the strictly lower triangular
 *          part of A is not referenced.
 *          If uplo = 'L', the leading N-by-N lower triangular part of A contains the lower
 *          triangular part of the matrix A, and the strictly upper triangular part of A is not
 *          referenced.
 *          On exit, if return value = 0, the factor U or L from the Cholesky factorization
 *          A = U**H*U or A = L*L**H.
 *
 * @param[in] lda
 *          The leading dimension of the array A. lda >= max(1,n).
 *
 *******************************************************************************/
void CORE_zpotrf(PLASMA_enum uplo, int n,
                     PLASMA_Complex64_t *A, int lda)
{
    LAPACKE_zpotrf(
        LAPACK_COL_MAJOR,
        lapack_const(uplo),
        n, A, lda);
}

/******************************************************************************/
void CORE_OMP_zpotrf(PLASMA_enum uplo, int n,
                     PLASMA_Complex64_t *A, int lda)
{
#pragma omp task depend(inout:A[0:n*n])
    CORE_zpotrf(
        uplo, n,
        A, lda);
}
