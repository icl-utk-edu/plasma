/**
 *
 * @file core_zherk.c
 *
 *  PLASMA core_blas kernel.
 *  PLASMA is a software package provided by Univ. of Tennessee,
 *  Univ. of California Berkeley, Univ. of Colorado Denver and
 *  Univ. of Manchester.
 *
 * @version 3.0.0
 * @author Pedro V. Lara
 * @date 2016-05-24
 * @precisions normal z -> c
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
 * @ingroup CORE_PLASMA_Complex64_t
 *
 *  Performs one of the hermitian rank k operations
 *
 *    \f[ C = \alpha A \times A^H + \beta C \f],
 *    or
 *    \f[ C = \alpha A^H \times A + \beta C \f],
 *
 *  where op( X ) is one of:
 *          - op( X ) = X  or 
 *          - op( X ) = conjg( X' )
 *
 *  alpha and beta are real scalars, C is an n-by-n hermitian
 *  matrix and A is an n-by-k matrix in the first case and a k-by-n
 *  matrix in the second case.
 *
 *******************************************************************************
 *
 * @param[in] uplo
 *          - PlasmaUpper: Upper triangle of C is stored;
 *          - PlasmaLower: Lower triangle of C is stored.
 *
 * @param[in] trans
 *          - PlasmaNoTrans:   A is not transposed;
 *          - PlasmaConjTrans  :   A is conjugate transposed.
 *
 * @param[in] n
 *          The order of the matrix C. n >= 0.
 *
 * @param[in] k
 *          The number of columns of the matrix op( A ).
 *
 * @param[in] alpha
 *          The scalar alpha.
 *
 * @param[in] A
 *          A is a lda-by-ka matrix, where ka is k when trans = PlasmaNoTrans,
 *          and is n otherwise.
 *
 * @param[in] lda
 *          The leading dimension of the array A. lda must be at least
 *          max(1, n) if trans == PlasmaNoTrans, otherwise lda must
 *          be at least max(1, k).
 *
 * @param[in] beta
 *          beta specifies the scalar beta
 *
 * @param[in,out] C
 *          C is a ldc-by-n matrix.
 *          On exit, the array uplo part of the matrix is overwritten
 *          by the uplo part of the updated matrix.
 *
 * @param[in] ldc
 *          The leading dimension of the array C. ldc >= max(1, n).
 *
 *******************************************************************************/
void CORE_zherk(
    PLASMA_enum uplo, PLASMA_enum trans, 
    int n, int k,
    double alpha, 
    const PLASMA_Complex64_t *A, int lda,
    double beta,  
    PLASMA_Complex64_t *C, int ldc)
{

    cblas_zherk(
        CblasColMajor,
        (CBLAS_UPLO)uplo, (CBLAS_TRANSPOSE)trans,
        n, k,
        alpha, A, lda,
        beta, C, ldc);
}

/******************************************************************************/

void CORE_OMP_zherk(
    PLASMA_enum uplo, PLASMA_enum trans, 
    int n, int k,
    double alpha, 
    const PLASMA_Complex64_t *A, int lda,
    double beta,  
    PLASMA_Complex64_t *C, int ldc)
{
    // omp depends assume lda == n or k, and ldc == n,
    // depending on transposes
#pragma omp task depend(in:A[0:n*k]) depend(inout:C[0:n*n])
    CORE_zherk(
        uplo, trans,
        n, k,
        alpha, A, lda,
        beta, C, ldc);
}
