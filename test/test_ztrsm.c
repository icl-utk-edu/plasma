/**
 *
 * @file testing_ztrsm.c
 *
 *  PLASMA testing routines
 *  PLASMA is a software package provided by Univ. of Tennessee,
 *  Univ. of California Berkeley and Univ. of Colorado Denver
 *
 * @version 3.0.0
 * @author Mawussi Zounon
 * @precisions normal z -> c d s
 *
 **/
#include "test.h"
#include "flops.h"

#include <assert.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef PLASMA_WITH_MKL
    #include <mkl_cblas.h>
    #include <mkl_lapacke.h>
#else
    #include <cblas.h>
    #include <lapacke.h>
#endif
#include <omp.h>
#include <plasma.h>

#define COMPLEX

#define A(i_, j_)  (A + (i_) + (size_t)lda*(j_))

/***************************************************************************//**
 *
 * @brief Tests ZTRSM.
 *
 * @param[in]  param - array of parameters
 * @param[out] info  - string of column labels or column values; length InfoLen
 *
 * If param is NULL     and info is NULL,     print usage and return.
 * If param is NULL     and info is non-NULL, set info to column headings and return.
 * If param is non-NULL and info is non-NULL, set info to column values   and run test.
 ******************************************************************************/
void test_ztrsm(param_value_t param[], char *info)
{
    //================================================================
    // Print usage info or return column labels or values.
    //================================================================
    if (param == NULL) {
        if (info == NULL) {
            // Print usage info.
            print_usage(PARAM_SIDE);
            print_usage(PARAM_UPLO);
            print_usage(PARAM_TRANSA);
            print_usage(PARAM_DIAG);
            print_usage(PARAM_M);
            print_usage(PARAM_N);
            print_usage(PARAM_ALPHA);
            print_usage(PARAM_PADA);
            print_usage(PARAM_PADB);
        } else {
            // Return column labels.
            snprintf(info, InfoLen,
                     "%*s %*s %*s %*s %*s %*s %*s %*s %*s",
                     InfoSpacing, "side",
                     InfoSpacing, "uplo",
                     InfoSpacing, "TransA",
                     InfoSpacing, "diag",
                     InfoSpacing, "M",
                     InfoSpacing, "N",
                     InfoSpacing, "alpha",
                     InfoSpacing, "PadA",
                     InfoSpacing, "PadB");
        }
        return;
    }
    // Return column values.
    snprintf(info, InfoLen,
             "%*c %*c %*c %*c %*d %*d %*.4f %*d %*d",
             InfoSpacing, param[PARAM_SIDE].c,
             InfoSpacing, param[PARAM_UPLO].c,
             InfoSpacing, param[PARAM_TRANSA].c,
             InfoSpacing, param[PARAM_DIAG].c,
             InfoSpacing, param[PARAM_M].i,
             InfoSpacing, param[PARAM_N].i,
             InfoSpacing, __real__(param[PARAM_ALPHA].z),
             InfoSpacing, param[PARAM_PADA].i,
             InfoSpacing, param[PARAM_PADB].i);

    //================================================================
    // Set parameters.
    //================================================================
    PLASMA_enum side;
    PLASMA_enum uplo;
    PLASMA_enum transa;
    PLASMA_enum diag;

    if (param[PARAM_SIDE].c == 'l')
        side = PlasmaLeft;
    else
        side = PlasmaRight;

    if (param[PARAM_UPLO].c == 'l')
        uplo = PlasmaLower;
    else
        uplo = PlasmaUpper;

    if (param[PARAM_TRANSA].c == 'n')
        transa = PlasmaNoTrans;
    else if (param[PARAM_TRANSA].c == 't')
        transa = PlasmaTrans;
    else
        transa = PlasmaConjTrans;

    if (param[PARAM_DIAG].c == 'n')
        diag = PlasmaNonUnit;
    else
        diag = PlasmaUnit;

    int m = param[PARAM_M].i;
    int n = param[PARAM_N].i;
    int Am;

    if (side == PlasmaLeft) {
        Am = m;
    }
    else {
        Am = n;
    }

    int lda = imax(1, Am + param[PARAM_PADA].i);
    int ldb = imax(1, m + param[PARAM_PADB].i);

    int test = param[PARAM_TEST].c == 'y';
    double tol = param[PARAM_TOL].d * LAPACKE_dlamch('E');

    //================================================================
    // Allocate and initialize arrays.
    //================================================================
    PLASMA_Complex64_t *A =
        (PLASMA_Complex64_t*)malloc((size_t)lda*lda*sizeof(PLASMA_Complex64_t));
    assert(A != NULL);

    PLASMA_Complex64_t *B =
        (PLASMA_Complex64_t*)malloc((size_t)ldb*n*sizeof(PLASMA_Complex64_t));
    assert(B != NULL);

    int seed[] = {0, 0, 0, 1};
    lapack_int retval;

    //=================================================================
    // Initialize the matrices.
    // Factor A into LU to get well-conditioned triangular matrix.
    // Copy L to U, since L seems okay when used with non-unit diagonal
    // (i.e., from U), while U fails when used with unit diagonal.
    //=================================================================
    retval = LAPACKE_zlarnv(1, seed, (size_t)lda*lda, A);
    assert(retval == 0);

    int ipiv[lda];
    LAPACKE_zgetrf(CblasColMajor, Am, Am, A, lda, ipiv);

    for(int j = 0; j < Am; j++)
        for(int i = 0; i < j; i++)
	  *A(i,j) = *A(j,i);

    retval = LAPACKE_zlarnv(1, seed, (size_t)ldb*n, B);
    assert(retval == 0);

    PLASMA_Complex64_t *Bref = NULL;
    if (test) {
        Bref = (PLASMA_Complex64_t*)malloc(
            (size_t)ldb*n*sizeof(PLASMA_Complex64_t));
        assert(Bref != NULL);

        memcpy(Bref, B, (size_t)ldb*n*sizeof(PLASMA_Complex64_t));
    }

#ifdef COMPLEX
    PLASMA_Complex64_t alpha = param[PARAM_ALPHA].z;
#else
    PLASMA_Complex64_t alpha = __real__(param[PARAM_ALPHA].z);
#endif

    //================================================================
    // Run and time PLASMA.
    //================================================================
    plasma_time_t start = omp_get_wtime();

    PLASMA_ztrsm(
        (CBLAS_SIDE)side, (CBLAS_UPLO) uplo,
        (CBLAS_TRANSPOSE)transa, (CBLAS_DIAG)diag,
        m, n,
        alpha, A, lda,
               B, ldb);

    plasma_time_t stop = omp_get_wtime();
    plasma_time_t time = stop-start;

    param[PARAM_TIME].d = time;
    param[PARAM_GFLOPS].d = flops_ztrsm(side, m, n) / time / 1e9;

    //================================================================
    // Test results by comparing to a reference implementation.
    //================================================================
    if (test) {
        cblas_ztrsm(
            CblasColMajor,
            (CBLAS_SIDE)side, (CBLAS_UPLO)uplo,
            (CBLAS_TRANSPOSE)transa, (CBLAS_DIAG)diag,
            m, n,
            CBLAS_SADDR(alpha), A, lda,
            Bref, ldb);

        PLASMA_Complex64_t zmone = -1.0;
        cblas_zaxpy((size_t)ldb*n, CBLAS_SADDR(zmone), Bref, 1, B, 1);

        double work[1];
        double Bnorm = LAPACKE_zlange_work(
                           LAPACK_COL_MAJOR, 'F', m, n, Bref, ldb, work);
        double error = LAPACKE_zlange_work(
                           LAPACK_COL_MAJOR, 'F', m, n, B,    ldb, work);
        if (Bnorm != 0)
            error /= Bnorm;
        param[PARAM_ERROR].d = error;
        param[PARAM_SUCCESS].i = error < tol*n;
    }

    //================================================================
    // Free arrays.
    //================================================================
    free(A);
    free(B);
    if (test)
        free(Bref);
}
