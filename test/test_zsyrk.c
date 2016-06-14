/**
 *
 * @file test_zsyrk.c
 *
 *  PLASMA test routine.
 *  PLASMA is a software package provided by Univ. of Tennessee,
 *  Univ. of California Berkeley, Univ. of Colorado Denver and
 *  Univ. of Manchester.
 *
 * @version 3.0.0
 * @author Pedro V. Lara
 * @date 2016-05-12
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
 * If param is NULL     and info is NULL,     print usage and return.
 * If param is NULL     and info is non-NULL, set info to column headings and return.
 * If param is non-NULL and info is non-NULL, set info to column values   and run test.
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
        }
        else {
            // Return column labels.
            snprintf(info, InfoLen,
                "%*s %*s %*s %*s %*s %*s %*s %*s",
                InfoSpacing, "Uplo",
                InfoSpacing, "Trans",
                InfoSpacing, "N",
                InfoSpacing, "K",
                InfoSpacing, "alpha",
                InfoSpacing, "beta",
                InfoSpacing, "PadA",
                InfoSpacing, "PadC");
        }
        return;
    }
    // Return column values.
    snprintf(info, InfoLen,
        "%*c %*c %*d %*d %*.4f %*.4f %*d %*d",
        InfoSpacing, param[PARAM_UPLO].c,
        InfoSpacing, param[PARAM_TRANS].c,
        InfoSpacing, param[PARAM_N].i,
        InfoSpacing, param[PARAM_K].i,
        InfoSpacing, __real__(param[PARAM_ALPHA].z),
        InfoSpacing, __real__(param[PARAM_BETA].z),
        InfoSpacing, param[PARAM_PADA].i,
        InfoSpacing, param[PARAM_PADC].i);

    //================================================================
    // Set parameters.
    //================================================================
    PLASMA_enum uplo;
    PLASMA_enum trans;

    if (param[PARAM_UPLO].c == 'l')
        uplo = PlasmaLower;
    else 
        uplo = PlasmaUpper;
    
    if (param[PARAM_TRANS].c == 'n')
        trans = PlasmaNoTrans;
    else if (param[PARAM_TRANS].c == 't')
        trans = PlasmaTrans;
    else
        trans = PlasmaConjTrans;

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
    double tol = param[PARAM_TOL].d * LAPACKE_dlamch('E');

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
    PLASMA_Complex64_t alpha = __real__(param[PARAM_ALPHA].z);
    PLASMA_Complex64_t beta  = __real__(param[PARAM_BETA].z);
    #endif

    //================================================================
    // Run and time PLASMA.
    //================================================================
    plasma_time_t start = omp_get_wtime();
    PLASMA_zsyrk(
        (CBLAS_UPLO)uplo, (CBLAS_TRANSPOSE)trans,
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
        cblas_zsyrk(
            CblasColMajor,
            (CBLAS_UPLO)uplo, (CBLAS_TRANSPOSE)trans,
            n, k,
            CBLAS_SADDR(alpha), A, lda,
            CBLAS_SADDR(beta), Cref, ldc);

        PLASMA_Complex64_t zmone = -1.0;
        cblas_zaxpy((size_t)ldc*Cn, CBLAS_SADDR(zmone), Cref, 1, C, 1);

        double work[1];
        double Cnorm = LAPACKE_zlange_work(
                           LAPACK_COL_MAJOR, 'F', Cm, Cn, Cref, ldc, work);
        double error = LAPACKE_zlange_work(
                           LAPACK_COL_MAJOR, 'F', Cm, Cn, C,    ldc, work);
        if (Cnorm != 0)
            error /= Cnorm;

        param[PARAM_ERROR].d = error;
        param[PARAM_SUCCESS].i = error < tol;
    }

    //================================================================
    // Free arrays.
    //================================================================
    free(A);
    free(C);
    if (test)
        free(Cref);
}
