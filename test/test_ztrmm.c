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

/***************************************************************************//**
 *
 * @brief Tests ZTRMM
 *
 * @param[in]  param - array of parameters
 * @param[out] info  - string of column labels or column values; length InfoLen
 *
 * If param is NULL     and info is NULL,     print usage and return.
 * If param is NULL     and info is non-NULL, set info to column headings
 * and return.
 * If param is non-NULL and info is non-NULL, set info to column values
 * and run test.
 ******************************************************************************/
void test_ztrmm(param_value_t param[], char *info)
{
    //================================================================
    // Print usage info or return column labels or values
    //================================================================
    if (param == NULL) {
        if (info == NULL) {
            // Print usage info
            print_usage(PARAM_SIDE);
            print_usage(PARAM_UPLO);
            print_usage(PARAM_TRANSA);
            print_usage(PARAM_DIAG);
            print_usage(PARAM_M);
            print_usage(PARAM_N);
            print_usage(PARAM_ALPHA);
            print_usage(PARAM_PADA);
            print_usage(PARAM_PADB);
        }
        else {
            // Return column labels
            snprintf(info, InfoLen,
                "%*s %*s %*s %*s %*s %*s %*s %*s %*s",
                InfoSpacing, "Side",
                InfoSpacing, "UpLo",
                InfoSpacing, "TransA",
                InfoSpacing, "Diag",
                InfoSpacing, "m",
                InfoSpacing, "n",
                InfoSpacing, "alpha",
                InfoSpacing, "PadA",
                InfoSpacing, "PadB");
        }
        return;
    }
    // Return column values
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
    // Set parameters
    //================================================================
    PLASMA_enum side = PLASMA_side_const(param[PARAM_SIDE].c);
    PLASMA_enum uplo = PLASMA_uplo_const(param[PARAM_UPLO].c);
    PLASMA_enum transa = PLASMA_trans_const(param[PARAM_TRANSA].c);
    PLASMA_enum diag = PLASMA_diag_const(param[PARAM_DIAG].c);

    int m = param[PARAM_M].i;
    int n = param[PARAM_N].i;

    int k;
    int lda;

    if (side == PlasmaLeft) {
        k    = m;
        lda  = imax(1, m + param[PARAM_PADA].i);
    }
    else {
        k    = n;
        lda  = imax(1, n + param[PARAM_PADA].i);
    }

    int    ldb  = imax(1, m + param[PARAM_PADB].i);
    int    test = param[PARAM_TEST].c == 'y';
    double tol  = param[PARAM_TOL].d * LAPACKE_dlamch('E');

#ifdef COMPLEX
    PLASMA_Complex64_t alpha = param[PARAM_ALPHA].z;
#else
    PLASMA_Complex64_t alpha = __real__(param[PARAM_ALPHA].z);
#endif

    //================================================================
    // Allocate and initialize arrays
    //================================================================
    PLASMA_Complex64_t *A =
        (PLASMA_Complex64_t *)malloc((size_t)lda*k*sizeof(PLASMA_Complex64_t));
    assert(A != NULL);

    PLASMA_Complex64_t *B =
        (PLASMA_Complex64_t *)malloc((size_t)ldb*n*sizeof(PLASMA_Complex64_t));
    assert(B != NULL);

    int seed[] = {0, 0, 0, 1};
    lapack_int retval;
    retval = LAPACKE_zlarnv(1, seed, (size_t)lda*k, A);
    assert(retval == 0);

    retval = LAPACKE_zlarnv(1, seed, (size_t)ldb*n, B);
    assert(retval == 0);

    PLASMA_Complex64_t *Bref = NULL;
    if (test) {
        Bref = (PLASMA_Complex64_t*)malloc(
            (size_t)ldb*n*sizeof(PLASMA_Complex64_t));
        assert(Bref != NULL);

        memcpy(Bref, B, (size_t)ldb*n*sizeof(PLASMA_Complex64_t));
    }

    //================================================================
    // Run and time PLASMA
    //================================================================
    plasma_time_t start = omp_get_wtime();

    PLASMA_ztrmm(side, uplo,
                 transa, diag,
                 m, n, alpha, A, lda, B, ldb);

    plasma_time_t stop = omp_get_wtime();
    plasma_time_t time = stop-start;

    param[PARAM_TIME].d   = time;
    param[PARAM_GFLOPS].d = flops_ztrmm(side, m, n) / time / 1e9;

    //================================================================
    // Test results by comparing to a reference implementation
    //================================================================
    if (test) {
        cblas_ztrmm(CblasColMajor, (CBLAS_SIDE)side, (CBLAS_UPLO)uplo,
                   (CBLAS_TRANSPOSE)transa, (CBLAS_DIAG)diag,
                    m, n, CBLAS_SADDR(alpha), A, lda, Bref, ldb);

        PLASMA_Complex64_t zmone = -1.0;

        k = ldb > m ? ldb : m;

        cblas_zaxpy((size_t)k*n, CBLAS_SADDR(zmone), Bref, 1, B, 1);

        double work[1];
        double Bnorm = LAPACKE_zlange_work(
                           LAPACK_COL_MAJOR, 'F', m, n, Bref, ldb, work);
        double error = LAPACKE_zlange_work(
                           LAPACK_COL_MAJOR, 'F', m, n, B,    ldb, work);
        if (Bnorm != 0)
            error /= Bnorm;

        param[PARAM_ERROR].d   = error;
        param[PARAM_SUCCESS].i = error < tol;
    }

    //================================================================
    // Free arrays
    //================================================================
    free(A);
    free(B);
    if (test)
        free(Bref);
}
