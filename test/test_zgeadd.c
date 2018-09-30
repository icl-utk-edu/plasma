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
#include "plasma.h"
#include <plasma_core_blas.h>
#include "core_lapack.h"

#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <omp.h>

#define COMPLEX

/***************************************************************************//**
 *
 * @brief Tests ZGEADD
 *
 * @param[in,out] param - array of parameters
 * @param[in]     run - whether to run test
 *
 * Sets flags in param indicating which parameters are used.
 * If run is true, also runs test and stores output parameters.
 ******************************************************************************/
void test_zgeadd(param_value_t param[], bool run)
{
    //================================================================
    // Mark which parameters are used.
    //================================================================
    param[PARAM_TRANSA ].used = true;
    param[PARAM_DIM    ].used = PARAM_USE_M | PARAM_USE_N;
    param[PARAM_ALPHA  ].used = true;
    param[PARAM_BETA   ].used = true;
    param[PARAM_PADA   ].used = true;
    param[PARAM_PADB   ].used = true;
    param[PARAM_NB     ].used = true;
    if (! run)
        return;

    //================================================================
    // Set parameters
    //================================================================
    plasma_enum_t transa = plasma_trans_const(param[PARAM_TRANSA].c);

    int m = param[PARAM_DIM].dim.m;
    int n = param[PARAM_DIM].dim.n;

    int Am, An;
    int Bm, Bn;

    if (transa == PlasmaNoTrans) {
        Am = m;
        An = n;
    }
    else {
        Am = n;
        An = m;
    }

    Bm = m;
    Bn = n;

    int lda = imax(1, Am + param[PARAM_PADA].i);
    int ldb = imax(1, Bm + param[PARAM_PADB].i);

    int    test = param[PARAM_TEST].c == 'y';
    double eps  = LAPACKE_dlamch('E');

    //================================================================
    // Set tuning parameters
    //================================================================
    plasma_set(PlasmaTuning, PlasmaDisabled);
    plasma_set(PlasmaNb, param[PARAM_NB].i);

    //================================================================
    // Allocate and initialize arrays
    //================================================================
    plasma_complex64_t *A =
        (plasma_complex64_t*)malloc((size_t)lda*An*sizeof(plasma_complex64_t));
    assert(A != NULL);

    plasma_complex64_t *B =
        (plasma_complex64_t*)malloc((size_t)ldb*Bn*sizeof(plasma_complex64_t));
    assert(B != NULL);

    int seed[] = {0, 0, 0, 1};
    lapack_int retval;
    retval = LAPACKE_zlarnv(1, seed, (size_t)lda*An, A);
    assert(retval == 0);

    retval = LAPACKE_zlarnv(1, seed, (size_t)ldb*Bn, B);
    assert(retval == 0);

    plasma_complex64_t *Bref = NULL;
    if (test) {
        Bref = (plasma_complex64_t*)malloc(
            (size_t)ldb*Bn*sizeof(plasma_complex64_t));
        assert(Bref != NULL);

        memcpy(Bref, B, (size_t)ldb*Bn*sizeof(plasma_complex64_t));
    }

#ifdef COMPLEX
    plasma_complex64_t alpha = param[PARAM_ALPHA].z;
    plasma_complex64_t beta  = param[PARAM_BETA].z;
#else
    double alpha = creal(param[PARAM_ALPHA].z);
    double beta  = creal(param[PARAM_BETA].z);
#endif

    //================================================================
    // Run and time PLASMA
    //================================================================
    plasma_time_t start = omp_get_wtime();

    retval = plasma_zgeadd(transa, m, n, alpha, A, lda, beta, B, ldb);

    plasma_time_t stop = omp_get_wtime();

    if (retval != PlasmaSuccess) {
        plasma_error("plasma_zgeadd() failed");
        param[PARAM_TIME].d    = 0.0;
        param[PARAM_GFLOPS].d  = 0.0;
        param[PARAM_ERROR].d   = 1.0;
        param[PARAM_SUCCESS].i = false;
        return;
    }
    else {
        plasma_time_t time    = stop-start;
        param[PARAM_TIME].d   = time;
        param[PARAM_GFLOPS].d = flops_zgeadd(m, n) / time / 1e9;
    }

    //==================================================================
    // Test results by comparing to result of plasma_core_z{ge,tr}add function
    //==================================================================
    if (test) {
        // Calculate relative error |B_ref - B|_F / |B_ref|_F < 3*eps
        // Using 3*eps covers complex arithmetic

        switch (transa) {
        //=================
        // PlasmaConjTrans
        //=================
        case PlasmaConjTrans:
            for (int j = 0; j < n; j++) {
                for (int i = 0; i < m; i++) {
                    Bref[ldb*j+i] =
                        beta * Bref[ldb*j+i] + alpha * conj(A[lda*i+j]);
                }
            }
            break;
        //=================
        // PlasmaTrans
        //=================
        case PlasmaTrans:
            for (int j = 0; j < n; j++) {
                for (int i = 0; i < m; i++) {
                    Bref[ldb*j+i] =
                        beta * Bref[ldb*j+i] + alpha * A[lda*i+j];
                }
            }
            break;
        //=================
        // PlasmaNoTrans
        //=================
        case PlasmaNoTrans:
            for (int j = 0; j < n; j++) {
                for (int i = 0; i < m; i++) {
                    Bref[ldb*j+i] =
                        beta * Bref[ldb*j+i] + alpha * A[lda*j+i];
                }
            }
        }

        double work[1];

        // Calculate Frobenius norm of reference result B_ref
        double BnormRef  = LAPACKE_zlange_work(
                               LAPACK_COL_MAJOR, 'F', Bm, Bn, Bref, ldb, work);

        // Calculate difference B_ref-B
        plasma_complex64_t zmone = -1.0;
        cblas_zaxpy((size_t)ldb*Bn, CBLAS_SADDR(zmone), B, 1, Bref, 1);

        // Calculate Frobenius norm of B_ref-B
        double BnormDiff = LAPACKE_zlange_work(
                               LAPACK_COL_MAJOR, 'F', Bm, Bn, Bref, ldb, work);

        // Calculate relative error |B_ref-B|_F / |B_ref|_F
        double error = BnormDiff/BnormRef;

        param[PARAM_ERROR].d   = error;
        param[PARAM_SUCCESS].i = error < 3*eps;
    }

    //================================================================
    // Free arrays
    //================================================================
    free(A);
    free(B);

    if (test)
        free(Bref);
}
