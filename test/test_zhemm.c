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
#include "core_lapack.h"
#include "plasma.h"

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
 * @brief Tests ZHEMM.
 *
 * @param[in]  param - array of parameters
 * @param[out] info  - string of column labels or column values; length InfoLen
 *
 * If param is NULL and info is NULL,     print usage and return.
 * If param is NULL and info is non-NULL, set info to column labels and return.
 * If param is non-NULL and info is non-NULL, set info to column values
 * and run test.
 ******************************************************************************/
void test_zhemm(param_value_t param[], char *info)
{
    //================================================================
    // Print usage info or return column labels or values.
    //================================================================
    if (param == NULL) {
        if (info == NULL) {
            // Print usage info.
            print_usage(PARAM_SIDE);
            print_usage(PARAM_UPLO);
            print_usage(PARAM_M);
            print_usage(PARAM_N);
            print_usage(PARAM_ALPHA);
            print_usage(PARAM_BETA);
            print_usage(PARAM_PADA);
            print_usage(PARAM_PADB);
            print_usage(PARAM_PADC);
            print_usage(PARAM_NB);
        }
        else {
            // Return column labels.
            snprintf(info, InfoLen,
                     "%*s %*s %*s %*s %*s %*s %*s %*s %*s %*s",
                     InfoSpacing, "side",
                     InfoSpacing, "uplo",
                     InfoSpacing, "m",
                     InfoSpacing, "n",
                     InfoSpacing, "alpha",
                     InfoSpacing, "beta",
                     InfoSpacing, "PadA",
                     InfoSpacing, "PadB",
                     InfoSpacing, "PadC",
                     InfoSpacing, "NB");
        }
        return;
    }
    // Return column values.
    snprintf(info, InfoLen,
             "%*c %*c %*d %*d %*.4f %*.4f %*d %*d %*d %*d",
             InfoSpacing, param[PARAM_SIDE].c,
             InfoSpacing, param[PARAM_UPLO].c,
             InfoSpacing, param[PARAM_M].i,
             InfoSpacing, param[PARAM_N].i,
             InfoSpacing, creal(param[PARAM_ALPHA].z),
             InfoSpacing, creal(param[PARAM_BETA].z),
             InfoSpacing, param[PARAM_PADA].i,
             InfoSpacing, param[PARAM_PADB].i,
             InfoSpacing, param[PARAM_PADC].i,
             InfoSpacing, param[PARAM_NB].i);

    //================================================================
    // Set parameters.
    //================================================================
    plasma_enum_t side = plasma_side_const(param[PARAM_SIDE].c);
    plasma_enum_t uplo = plasma_uplo_const(param[PARAM_UPLO].c);

    int m = param[PARAM_M].i;
    int n = param[PARAM_N].i;

    int Am, An;
    int Bm, Bn;
    int Cm, Cn;

    if (side == PlasmaLeft) {
        Am = m;
        An = m;
    }
    else {
        Am = n;
        An = n;
    }
    Bm = m;
    Bn = n;

    Cm = m;
    Cn = n;

    int lda = imax(1, Am + param[PARAM_PADA].i);
    int ldb = imax(1, Bm + param[PARAM_PADB].i);
    int ldc = imax(1, Cm + param[PARAM_PADC].i);

    int test = param[PARAM_TEST].c == 'y';
    double eps = LAPACKE_dlamch('E');

#ifdef COMPLEX
    plasma_complex64_t alpha = param[PARAM_ALPHA].z;
    plasma_complex64_t beta  = param[PARAM_BETA].z;
#else
    double alpha = creal(param[PARAM_ALPHA].z);
    double beta  = creal(param[PARAM_BETA].z);
#endif

    //================================================================
    // Set tuning parameters.
    //================================================================
    plasma_set(PlasmaNb, param[PARAM_NB].i);

    //================================================================
    // Allocate and initialize arrays.
    //================================================================
    plasma_complex64_t *A =
        (plasma_complex64_t*)malloc((size_t)lda*An*sizeof(plasma_complex64_t));
    assert(A != NULL);

    plasma_complex64_t *B =
        (plasma_complex64_t*)malloc((size_t)ldb*Bn*sizeof(plasma_complex64_t));
    assert(B != NULL);

    plasma_complex64_t *C =
        (plasma_complex64_t*)malloc((size_t)ldc*Cn*sizeof(plasma_complex64_t));
    assert(C != NULL);

    int seed[] = {0, 0, 0, 1};
    lapack_int retval;
    retval = LAPACKE_zlarnv(1, seed, (size_t)lda*An, A);
    assert(retval == 0);

    retval = LAPACKE_zlarnv(1, seed, (size_t)ldb*Bn, B);
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

    plasma_zhemm(
        side, uplo,
        m, n,
        alpha, A, lda,
               B, ldb,
        beta,  C, ldc);

    plasma_time_t stop = omp_get_wtime();
    plasma_time_t time = stop-start;

    param[PARAM_TIME].d = time;
    param[PARAM_GFLOPS].d = flops_zhemm(side, m, n) / time / 1e9;

    //================================================================
    // Test results by comparing to a reference implementation.
    //================================================================
    if (test) {
        // see comments in test_zgemm.c
        char uplo_ = param[PARAM_UPLO].c;
        double work[1];
        double Anorm = LAPACKE_zlanhe_work(
                           LAPACK_COL_MAJOR, 'F', uplo_, An, A, lda, work);
        double Bnorm = LAPACKE_zlange_work(
                           LAPACK_COL_MAJOR, 'F', Bm, Bn, B,    ldb, work);
        double Cnorm = LAPACKE_zlange_work(
                           LAPACK_COL_MAJOR, 'F', Cm, Cn, Cref, ldc, work);

        cblas_zhemm(
            CblasColMajor,
            (CBLAS_SIDE) side, (CBLAS_UPLO) uplo,
            m, n,
            CBLAS_SADDR(alpha), A, lda,
                                B, ldb,
            CBLAS_SADDR(beta),  Cref, ldc);

        plasma_complex64_t zmone = -1.0;
        cblas_zaxpy((size_t)ldc*Cn, CBLAS_SADDR(zmone), Cref, 1, C, 1);

        double error = LAPACKE_zlange_work(
                           LAPACK_COL_MAJOR, 'F', Cm, Cn, C,    ldc, work);
        double normalize = sqrt((double)An+2) * cabs(alpha) * Anorm * Bnorm
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
    free(B);
    free(C);
    if (test)
        free(Cref);
}
