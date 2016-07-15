/**
 *
 * @file core_ztrmm.c
 *
 *  PLASMA core_blas kernel
 *  PLASMA is a software package provided by Univ. of Tennessee,
 *  Univ. of California Berkeley and Univ. of Colorado Denver
 *
 * @version 3.0.0
 * @author  Julien Langou
 * @author  Henricus Bouwmeester
 * @author  Mathieu Faverge
 * @author  Maksims Abalenkovs
 * @date    2016-06-22
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

/***************************************************************************/

void CORE_ztrmm(
    PLASMA_enum side, PLASMA_enum uplo, PLASMA_enum transA, PLASMA_enum diag,
    int m, int n,
    PLASMA_Complex64_t alpha, const PLASMA_Complex64_t *A, int lda,
                                    PLASMA_Complex64_t *B, int ldb)
{
    cblas_ztrmm(
        CblasColMajor,
        (CBLAS_SIDE)side, (CBLAS_UPLO)uplo,
        (CBLAS_TRANSPOSE)transA, (CBLAS_DIAG)diag,
        m, n,
        CBLAS_SADDR(alpha), A, lda,
        B, ldb);
}

/***************************************************************************/

void CORE_OMP_ztrmm(
    PLASMA_enum side, PLASMA_enum uplo, PLASMA_enum transA, PLASMA_enum diag,
    int m, int n,
    PLASMA_Complex64_t alpha, const PLASMA_Complex64_t *A, int lda,
                                    PLASMA_Complex64_t *B, int ldb)
{
    /* OpenMP depends assume lda == m or n, ldb == n or m depending on transposes */
#pragma omp task depend(in:A[0:m*n]) depend(inout:B[0:n*m])
    CORE_ztrmm(side, uplo,
               transA, diag,
               m, n,
               alpha, A, lda,
                      B, ldb);
}

/***************************************************************************/

void CORE_ztrmm_p2(
    PLASMA_enum side, PLASMA_enum uplo, PLASMA_enum transA, PLASMA_enum diag,
    int m, int n,
    PLASMA_Complex64_t alpha, const PLASMA_Complex64_t  *A, int lda,
                                    PLASMA_Complex64_t **B, int ldb)
{
    cblas_ztrmm(
        CblasColMajor,
        (CBLAS_SIDE)side, (CBLAS_UPLO)uplo,
        (CBLAS_TRANSPOSE)transA, (CBLAS_DIAG)diag,
        m, n,
        CBLAS_SADDR(alpha), A, lda,
        *B, ldb);
}

/***************************************************************************/

void CORE_OMP_ztrmm_p2(
    PLASMA_enum side, PLASMA_enum uplo, PLASMA_enum transA, PLASMA_enum diag,
    int m, int n,
    PLASMA_Complex64_t alpha, const PLASMA_Complex64_t  *A, int lda,
                                    PLASMA_Complex64_t **B, int ldb)
{
#pragma omp task depend(in:A[0:m*n]) depend(inout:B[0:n*m])
    CORE_ztrmm_p2(side,   uplo,
                  transA, diag,
                  m, n,
                  alpha, A, lda,
                         B, ldb);
}
