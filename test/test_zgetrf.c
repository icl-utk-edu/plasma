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
#include "core_blas.h"
#include "core_lapack.h"
#include "plasma.h"

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <omp.h>

#define COMPLEX

/***************************************************************************//**
 *
 * @brief Tests ZPOTRF.
 *
 * @param[in]  param - array of parameters
 * @param[out] info  - string of column labels or column values; length InfoLen
 *
 * If param is NULL and info is NULL,     print usage and return.
 * If param is NULL and info is non-NULL, set info to column labels and return.
 * If param is non-NULL and info is non-NULL, set info to column values
 * and run test.
 ******************************************************************************/
void test_zgetrf(param_value_t param[], char *info)
{
    //================================================================
    // Print usage info or return column labels or values.
    //================================================================
    if (param == NULL) {
        if (info == NULL) {
            // Print usage info.
            print_usage(PARAM_M);
            print_usage(PARAM_N);
            print_usage(PARAM_PADA);
            print_usage(PARAM_NB);
            print_usage(PARAM_IB);
            print_usage(PARAM_NTPF);
            print_usage(PARAM_ZEROCOL);
        }
        else {
            // Return column labels.
            snprintf(info, InfoLen,
                     "%*s %*s %*s %*s %*s %*s %*s",
                     InfoSpacing, "M",
                     InfoSpacing, "N",
                     InfoSpacing, "PadA",
                     InfoSpacing, "NB",
                     InfoSpacing, "IB",
                     InfoSpacing, "NTPF",
                     InfoSpacing, "ZeroCol");
        }
        return;
    }
    // Return column values.
    snprintf(info, InfoLen,
             "%*d %*d %*d %*d %*d %*d %*d",
             InfoSpacing, param[PARAM_M].i,
             InfoSpacing, param[PARAM_N].i,
             InfoSpacing, param[PARAM_PADA].i,
             InfoSpacing, param[PARAM_NB].i,
             InfoSpacing, param[PARAM_IB].i,
             InfoSpacing, param[PARAM_NTPF].i,
             InfoSpacing, param[PARAM_ZEROCOL].i);

    //================================================================
    // Set parameters.
    //================================================================
    int m = param[PARAM_M].i;
    int n = param[PARAM_N].i;

    int lda = imax(1, m+param[PARAM_PADA].i);

    int test = param[PARAM_TEST].c == 'y';
    double tol = param[PARAM_TOL].d * LAPACKE_dlamch('E');

    //================================================================
    // Set tuning parameters.
    //================================================================
    plasma_set(PlasmaNb, param[PARAM_NB].i);
    plasma_set(PlasmaIb, param[PARAM_IB].i);
    plasma_set(PlasmaNumPanelThreads, param[PARAM_NTPF].i);

    //================================================================
    // Allocate and initialize arrays.
    //================================================================
    plasma_complex64_t *A =
        (plasma_complex64_t*)malloc((size_t)lda*n*sizeof(plasma_complex64_t));
    assert(A != NULL);

    int *ipiv = (int*)malloc((size_t)m*sizeof(int));
    assert(ipiv != NULL);

    int seed[] = {0, 0, 0, 1};
    lapack_int retval;
    retval = LAPACKE_zlarnv(1, seed, (size_t)lda*n, A);
    assert(retval == 0);

    int zerocol = param[PARAM_ZEROCOL].i;
    if (zerocol >= 0 && zerocol < n)
        memset(&A[zerocol*lda], 0, m*sizeof(plasma_complex64_t));

    plasma_complex64_t *Aref = NULL;
    if (test) {
        Aref = (plasma_complex64_t*)malloc(
            (size_t)lda*n*sizeof(plasma_complex64_t));
        assert(Aref != NULL);

        memcpy(Aref, A, (size_t)lda*n*sizeof(plasma_complex64_t));
    }

    //================================================================
    // Run and time PLASMA.
    //================================================================
    plasma_time_t start = omp_get_wtime();
    int plainfo = plasma_zgetrf(m, n, A, lda, ipiv);
    plasma_time_t stop = omp_get_wtime();
    plasma_time_t time = stop-start;

    param[PARAM_TIME].d = time;
    param[PARAM_GFLOPS].d = flops_zgetrf(m, n) / time / 1e9;

    //================================================================
    // Test results by comparing to a reference implementation.
    // This will give spurious failures if LAPACK picks different pivots
    // than PLASMA. Should test solve or ||PA - LU||.
    //================================================================
    if (test) {
        int lapinfo = LAPACKE_zgetrf(
            LAPACK_COL_MAJOR,
            m, n,
            Aref, lda, ipiv);

        if (lapinfo == 0) {
            plasma_complex64_t zmone = -1.0;
            cblas_zaxpy((size_t)lda*n, CBLAS_SADDR(zmone), Aref, 1, A, 1);

            double work[1];
            double Anorm = LAPACKE_zlange_work(
                LAPACK_COL_MAJOR, 'F', m, n, Aref, lda, work);

            double error = LAPACKE_zlange_work(
                LAPACK_COL_MAJOR, 'F', m, n, A, lda, work);

            if (Anorm != 0.0)
                error /= Anorm;
            error /= sqrt((double)m*n);

            param[PARAM_ERROR].d = error;
            param[PARAM_SUCCESS].i = error < tol;
        }
        else {
            if (plainfo == lapinfo) {
                param[PARAM_ERROR].d = 0.0;
                param[PARAM_SUCCESS].i = 1;
            }
            else {
                param[PARAM_ERROR].d = INFINITY;
                param[PARAM_SUCCESS].i = 0;
            }
        }
    }

    //================================================================
    // Free arrays.
    //================================================================
    free(A);
    free(ipiv);
    if (test)
        free(Aref);
}
