/**
 *
 * @file core_zgemm.c
 *
 *  PLASMA core_blas kernel.
 *  PLASMA is a software package provided by Univ. of Tennessee,
 *  Univ. of California Berkeley and Univ. of Colorado Denver.
 *
 * @version 3.0.0
 * @author Jakub Kurzak
 * @date 2016-01-01
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
 * @ingroup core_gemm
 *
 *  Performs one of the matrix-matrix operations
 *
 *    \f[ C = \alpha [op( A )\times op( B )] + \beta C, \f]
 *
 *  where op( X ) is one of:
 *          - op( X ) = X   or
 *          - op( X ) = X^T or
 *          - op( X ) = X^H
 *
 *  alpha and beta are scalars, and A, B and C  are matrices, with op( A )
 *  an m-by-k matrix, op( B ) a k-by-n matrix and C an m-by-n matrix.
 *
 *******************************************************************************
 *
 * @param[in] transA
 *          - PlasmaNoTrans:   A is not transposed,
 *          - PlasmaTrans:     A is transposed,
 *          - PlasmaConjTrans: A is conjugate transposed.
 *
 * @param[in] transB
 *          - PlasmaNoTrans:   B is not transposed,
 *          - PlasmaTrans:     B is transposed,
 *          - PlasmaConjTrans: B is conjugate transposed.
 *
 * @param[in] m
 *          The number of rows of the matrix op( A ) and of the matrix C.
 *          m >= 0.
 *
 * @param[in] n
 *          The number of columns of the matrix op( B ) and of the matrix C.
 *          n >= 0.
 *
 * @param[in] k
 *          The number of columns of the matrix op( A ) and the number of rows
 *          of the matrix op( B ). k >= 0.
 *
 * @param[in] alpha
 *          The scalar alpha.
 *
 * @param[in] A
 *          An lda-by-ka matrix, where ka is k when transA = PlasmaNoTrans,
 *          and is m otherwise.
 *
 * @param[in] lda
 *          The leading dimension of the array A.
 *          When transA = PlasmaNoTrans, lda >= max(1,m),
 *          otherwise, lda >= max(1,k).
 *
 * @param[in] B
 *          An ldb-by-kb matrix, where kb is n when transB = PlasmaNoTrans,
 *          and is k otherwise.
 *
 * @param[in] ldb
 *          The leading dimension of the array B.
 *          When transB = PlasmaNoTrans, ldb >= max(1,k),
 *          otherwise, ldb >= max(1,n).
 *
 * @param[in] beta
 *          The scalar beta.
 *
 * @param[in,out] C
 *          An ldc-by-n matrix. On exit, the array is overwritten by the m-by-n
 *          matrix ( alpha*op( A )*op( B ) + beta*C ).
 *
 * @param[in] ldc
 *          The leading dimension of the array C. ldc >= max(1,m).
 *
 ******************************************************************************/
void CORE_zgemm(PLASMA_enum transA, PLASMA_enum transB,
                int m, int n, int k,
                PLASMA_Complex64_t alpha, const PLASMA_Complex64_t *A, int lda,
                                          const PLASMA_Complex64_t *B, int ldb,
                PLASMA_Complex64_t beta,        PLASMA_Complex64_t *C, int ldc)
{
    cblas_zgemm(CblasColMajor,
                (CBLAS_TRANSPOSE)transA, (CBLAS_TRANSPOSE)transB,
                m, n, k,
                CBLAS_SADDR(alpha), A, lda,
                                    B, ldb,
                CBLAS_SADDR(beta),  C, ldc);
}

/******************************************************************************/
void CORE_OMP_zgemm(
    PLASMA_enum transA, PLASMA_enum transB,
    int m, int n, int k,
    PLASMA_Complex64_t alpha, const PLASMA_Complex64_t *A, int lda,
                              const PLASMA_Complex64_t *B, int ldb,
    PLASMA_Complex64_t beta,        PLASMA_Complex64_t *C, int ldc)
{
    // omp depends assume lda == m or k, ldb == k or n, and ldc == m,
    // depending on transposes
    #pragma omp task depend(in:A[0:m*k]) \
                     depend(in:B[0:k*n]) \
                     depend(inout:C[0:m*n])
    CORE_zgemm(transA, transB,
               m, n, k,
               alpha, A, lda,
                      B, ldb,
               beta,  C, ldc);
}
