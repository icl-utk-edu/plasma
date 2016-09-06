/**
 *
 * @file
 *
 *  PLASMA is a software package provided by:
 *  University of Tennessee, US,
 *  University of Manchester, UK.
 *
 * @precisions normal z -> s d c
 *
 **/
#include "test.h"
#include "flops.h"

#include <assert.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

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
 * @brief Tests ZSYRK.
 *
 * @param[in]  param - array of parameters
 * @param[out] info  - string of column labels or column values; length InfoLen
 *
 * If param is NULL and info is NULL,     print usage and return.
 * If param is NULL and info is non-NULL, set info to column labels and return.
 * If param is non-NULL and info is non-NULL, set info to column values
 * and run test.
 ******************************************************************************/
void test_zsyrk(param_value_t param[], char *info)
{
    //================================================================
    // Print usage info or return column labels or values.
    //================================================================
    if (param == NULL) {
        if (info == NULL) {
            // Print usage info.
            print_usage(PARAM_UPLO);
            print_usage(PARAM_TRANS);
            print_usage(PARAM_N);
            print_usage(PARAM_K);
            print_usage(PARAM_ALPHA);
            print_usage(PARAM_BETA);
            print_usage(PARAM_PADA);
            print_usage(PARAM_PADC);
            print_usage(PARAM_NB);
        }
        else {
            // Return column labels.
            snprintf(info, InfoLen,
                     "%*s %*s %*s %*s %*s %*s %*s %*s %*s",
                     InfoSpacing, "Uplo",
                     InfoSpacing, "Trans",
                     InfoSpacing, "N",
                     InfoSpacing, "K",
                     InfoSpacing, "alpha",
                     InfoSpacing, "beta",
                     InfoSpacing, "PadA",
                     InfoSpacing, "PadC",
                     InfoSpacing, "NB");
        }
        return;
    }
    // Return column values.
    snprintf(info, InfoLen,
             "%*c %*c %*d %*d %*.4f %*.4f %*d %*d %*d",
             InfoSpacing, param[PARAM_UPLO].c,
             InfoSpacing, param[PARAM_TRANS].c,
             InfoSpacing, param[PARAM_N].i,
             InfoSpacing, param[PARAM_K].i,
             InfoSpacing, creal(param[PARAM_ALPHA].z),
             InfoSpacing, creal(param[PARAM_BETA].z),
             InfoSpacing, param[PARAM_PADA].i,
             InfoSpacing, param[PARAM_PADC].i,
             InfoSpacing, param[PARAM_NB].i);

    //================================================================
    // Set parameters.
    //================================================================
    PLASMA_enum uplo = PLASMA_uplo_const(param[PARAM_UPLO].c);
    PLASMA_enum trans = PLASMA_trans_const(param[PARAM_TRANS].c);

    int n = param[PARAM_N].i;
    int k = param[PARAM_K].i;

    int Am, An;
    int Cm, Cn;

    if (trans == PlasmaNoTrans) {
        Am = n;
        An = k;
    }
    else {
        Am = k;
        An = n;
    }
    Cm = n;
    Cn = n;

    int lda = imax(1, Am + param[PARAM_PADA].i);
    int ldc = imax(1, Cm + param[PARAM_PADC].i);

    int test = param[PARAM_TEST].c == 'y';
    double eps = LAPACKE_dlamch('E');

    //================================================================
    // Set tuning parameters.
    //================================================================
    PLASMA_Set(PLASMA_TILE_SIZE, param[PARAM_NB].i);

    //================================================================
    // Allocate and initialize arrays.
    //================================================================
    PLASMA_Complex64_t *A =
        (PLASMA_Complex64_t*)malloc((size_t)lda*An*sizeof(PLASMA_Complex64_t));
    assert(A != NULL);

    PLASMA_Complex64_t *C =
        (PLASMA_Complex64_t*)malloc((size_t)ldc*Cn*sizeof(PLASMA_Complex64_t));
    assert(C != NULL);

    int seed[] = {0, 0, 0, 1};
    lapack_int retval;
    retval = LAPACKE_zlarnv(1, seed, (size_t)lda*An, A);
    assert(retval == 0);

    retval = LAPACKE_zlarnv(1, seed, (size_t)ldc*Cn, C);
    assert(retval == 0);

    PLASMA_Complex64_t *Cref = NULL;
    if (test) {
        Cref = (PLASMA_Complex64_t*)malloc(
            (size_t)ldc*Cn*sizeof(PLASMA_Complex64_t));
        assert(Cref != NULL);

        memcpy(Cref, C, (size_t)ldc*Cn*sizeof(PLASMA_Complex64_t));
    }

#ifdef COMPLEX
    PLASMA_Complex64_t alpha = param[PARAM_ALPHA].z;
    PLASMA_Complex64_t beta  = param[PARAM_BETA].z;
#else
    double alpha = creal(param[PARAM_ALPHA].z);
    double beta  = creal(param[PARAM_BETA].z);
#endif

    //================================================================
    // Run and time PLASMA.
    //================================================================
    plasma_time_t start = omp_get_wtime();

    PLASMA_zsyrk(
        uplo, trans,
        n, k,
        alpha, A, lda,
        beta, C, ldc);

    plasma_time_t stop = omp_get_wtime();
    plasma_time_t time = stop-start;

    param[PARAM_TIME].d = time;
    param[PARAM_GFLOPS].d = flops_zsyrk(k, n) / time / 1e9;

    //================================================================
    // Test results by comparing to a reference implementation.
    //================================================================
    if (test) {
        // see comments in test_zgemm.c
        cblas_zsyrk(
            CblasColMajor,
            (CBLAS_UPLO)uplo, (CBLAS_TRANSPOSE)trans,
            n, k,
            CBLAS_SADDR(alpha), A, lda,
            CBLAS_SADDR(beta), Cref, ldc);

        PLASMA_Complex64_t zmone = -1.0;
        cblas_zaxpy((size_t)ldc*Cn, CBLAS_SADDR(zmone), Cref, 1, C, 1);

        char uplo_ = param[PARAM_UPLO].c;
        double work[1];
        double Anorm = LAPACKE_zlange_work(
                           LAPACK_COL_MAJOR, 'F', Am, An, A, lda, work);
        double Cnorm = LAPACKE_zlansy_work(
                           LAPACK_COL_MAJOR, 'F', uplo_, Cn, Cref, ldc, work);
        double error = LAPACKE_zlansy_work(
                           LAPACK_COL_MAJOR, 'F', uplo_, Cn, C,    ldc, work);
        double normalize = sqrt((double)k+2) * cabs(alpha) * Anorm * Anorm
                         + 2 * cabs(beta) * Cnorm;
        if (normalize != 0)
            error /= normalize;

        param[PARAM_ERROR].d = error;
        param[PARAM_SUCCESS].i = error < 3*eps;
    }

    //================================================================
    // Free arrays.
    //================================================================
    free(A);
    free(C);
    if (test)
        free(Cref);
}
