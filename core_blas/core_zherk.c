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

#include "core_blas.h"
#include "plasma_types.h"
#include "core_lapack.h"

/***************************************************************************//**
 *
 * @ingroup core_herk
 *
 *  Performs one of the Hermitian rank k operations
 *
 *    \f[ C = \alpha A \times A^H + \beta C, \f]
 *    or
 *    \f[ C = \alpha A^H \times A + \beta C, \f]
 *
 *  where alpha and beta are real scalars, C is an n-by-n Hermitian
 *  matrix, and A is an n-by-k matrix in the first case and a k-by-n
 *  matrix in the second case.
 *
 *******************************************************************************
 *
 * @param[in] uplo
 *          - PlasmaUpper: Upper triangle of C is stored;
 *          - PlasmaLower: Lower triangle of C is stored.
 *
 * @param[in] trans
 *          - PlasmaNoTrans:   \f[ C = \alpha A \times A^H + \beta C; \f]
 *          - PlasmaConjTrans: \f[ C = \alpha A^H \times A + \beta C. \f]
 *
 * @param[in] n
 *          The order of the matrix C. n >= 0.
 *
 * @param[in] k
 *          If trans = PlasmaNoTrans, number of columns of the A matrix;
 *          if trans = PlasmaConjTrans, number of rows of the A matrix.
 *
 * @param[in] alpha
 *          The scalar alpha.
 *
 * @param[in] A
 *          A is a lda-by-ka matrix.
 *          If trans = PlasmaNoTrans,   ka = k;
 *          if trans = PlasmaConjTrans, ka = n.
 *
 * @param[in] lda
 *          The leading dimension of the array A.
 *          If trans = PlasmaNoTrans,   lda >= max(1, n);
 *          if trans = PlasmaConjTrans, lda >= max(1, k).
 *
 * @param[in] beta
 *          The scalar beta.
 *
 * @param[in,out] C
 *          C is a ldc-by-n matrix.
 *          On exit, the uplo part of the matrix is overwritten
 *          by the uplo part of the updated matrix.
 *
 * @param[in] ldc
 *          The leading dimension of the array C. ldc >= max(1, n).
 *
 ******************************************************************************/
void CORE_zherk(PLASMA_enum uplo, PLASMA_enum trans,
                int n, int k,
                double alpha, const PLASMA_Complex64_t *A, int lda,
                double beta,        PLASMA_Complex64_t *C, int ldc)
{
    cblas_zherk(CblasColMajor,
                (CBLAS_UPLO)uplo, (CBLAS_TRANSPOSE)trans,
                n, k,
                alpha, A, lda,
                beta,  C, ldc);
}

/******************************************************************************/
void CORE_OMP_zherk(PLASMA_enum uplo, PLASMA_enum trans,
                    int n, int k,
                    double alpha, const PLASMA_Complex64_t *A, int lda,
                    double beta,        PLASMA_Complex64_t *C, int ldc)
{
    // omp depends assume lda == n or k, and ldc == n,
    // depending on trans.
    #pragma omp task depend(in:A[0:n*k]) depend(inout:C[0:n*n])
    CORE_zherk(uplo, trans,
               n, k,
               alpha, A, lda,
               beta,  C, ldc);
}
