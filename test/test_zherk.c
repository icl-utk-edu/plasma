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
#include "test.h"
#include "flops.h"
#include "plasma.h"
#include "core_lapack.h"

#include <assert.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <omp.h>

#define COMPLEX

/***************************************************************************//**
 *
 * @brief Tests ZHERK.
 *
 * @param[in,out] param - array of parameters
 * @param[in]     run - whether to run test
 *
 * Sets flags in param indicating which parameters are used.
 * If run is true, also runs test and stores output parameters.
 ******************************************************************************/
void test_zherk(param_value_t param[], bool run)
{
    //================================================================
    // Mark which parameters are used.
    //================================================================
    param[PARAM_UPLO   ].used = true;
    param[PARAM_TRANS  ].used = true;
    param[PARAM_DIM    ].used = PARAM_USE_N | PARAM_USE_K;
    param[PARAM_ALPHA  ].used = true;
    param[PARAM_BETA   ].used = true;
    param[PARAM_PADA   ].used = true;
    param[PARAM_PADC   ].used = true;
    param[PARAM_NB     ].used = true;
    if (! run)
        return;

    //================================================================
    // Set parameters.
    //================================================================
    plasma_enum_t uplo = plasma_uplo_const(param[PARAM_UPLO].c);
    plasma_enum_t trans = plasma_trans_const(param[PARAM_TRANS].c);

    int n = param[PARAM_DIM].dim.n;
    int k = param[PARAM_DIM].dim.k;

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

    double alpha = creal(param[PARAM_ALPHA].z);
    double beta  = creal(param[PARAM_BETA].z);

    //================================================================
    // Set tuning parameters.
    //================================================================
    plasma_set(PlasmaTuning, PlasmaDisabled);
    plasma_set(PlasmaNb, param[PARAM_NB].i);

    //================================================================
    // Allocate and initialize arrays.
    //================================================================
    plasma_complex64_t *A =
        (plasma_complex64_t*)malloc((size_t)lda*An*sizeof(plasma_complex64_t));
    assert(A != NULL);

    plasma_complex64_t *C =
        (plasma_complex64_t*)malloc((size_t)ldc*Cn*sizeof(plasma_complex64_t));
    assert(C != NULL);

    int seed[] = {0, 0, 0, 1};
    lapack_int retval;
    retval = LAPACKE_zlarnv(1, seed, (size_t)lda*An, A);
    assert(retval == 0);

    retval = LAPACKE_zlarnv(1, seed, (size_t)ldc*Cn, C);
    assert(retval == 0);

    plasma_complex64_t *Cref = NULL;
    if (test) {
        Cref = (plasma_complex64_t*)malloc(
            (size_t)ldc*Cn*sizeof(plasma_complex64_t));
        assert(Cref != NULL);

        memcpy(Cref, C, (size_t)ldc*Cn*sizeof(plasma_complex64_t));
    }

    //================================================================
    // Run and time PLASMA.
    //================================================================
    plasma_time_t start = omp_get_wtime();

    plasma_zherk(
        uplo, trans,
        n, k,
        alpha, A, lda,
        beta, C, ldc);

    plasma_time_t stop = omp_get_wtime();
    plasma_time_t time = stop-start;

    param[PARAM_TIME].d = time;
    param[PARAM_GFLOPS].d = flops_zherk(n, k) / time / 1e9;

    //================================================================
    // Test results by comparing to a reference implementation.
    //================================================================
    if (test) {
        // see comments in test_zgemm.c
        char uplo_ = param[PARAM_UPLO].c;
        double work[1];
        double Anorm = LAPACKE_zlange_work(
                           LAPACK_COL_MAJOR, 'F', Am, An, A, lda, work);
        double Cnorm = LAPACKE_zlanhe_work(
                           LAPACK_COL_MAJOR, 'F', uplo_, Cn, Cref, ldc, work);

        cblas_zherk(
            CblasColMajor,
            (CBLAS_UPLO)uplo, (CBLAS_TRANSPOSE)trans,
            n, k,
            alpha, A, lda,
            beta, Cref, ldc);

        plasma_complex64_t zmone = -1.0;
        cblas_zaxpy((size_t)ldc*Cn, CBLAS_SADDR(zmone), Cref, 1, C, 1);

        double error = LAPACKE_zlanhe_work(
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
