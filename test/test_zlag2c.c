/**
 *
 * @file
 *
 *  PLASMA is a software package provided by:
 *  University of Tennessee, US,
 *  University of Manchester, UK.
 *
 * @precisions mixed zc -> ds
 *
 **/

#include "flops.h"
#include "test.h"
#include "plasma.h"
#include <plasma_core_blas.h>
#include "core_lapack.h"

#include <assert.h>
#include <math.h>
#include <omp.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/***************************************************************************//**
 *
 * @brief Tests ZLAG2C
 *
 * @param[in,out] param - array of parameters
 * @param[in]     run - whether to run test
 *
 * Sets flags in param indicating which parameters are used.
 * If run is true, also runs test and stores output parameters.
 ******************************************************************************/
void test_zlag2c(param_value_t param[], bool run)
{
    //================================================================
    // Mark which parameters are used.
    //================================================================
    param[PARAM_DIM    ].used = PARAM_USE_M | PARAM_USE_N;
    param[PARAM_PADA   ].used = true;
    param[PARAM_PADB   ].used = true;
    param[PARAM_NB     ].used = true;
    if (! run)
        return;

    //================================================================
    // Set parameters
    //================================================================
    int m = param[PARAM_DIM].dim.m;
    int n = param[PARAM_DIM].dim.n;

    int lda  = imax(1, m + param[PARAM_PADA].i);
    int ldas = imax(1, m + param[PARAM_PADB].i);

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
        (plasma_complex64_t*)malloc((size_t)lda*n*sizeof(plasma_complex64_t));
    assert(A != NULL);

    plasma_complex32_t *As =
        (plasma_complex32_t*)malloc((size_t)ldas*n*sizeof(plasma_complex32_t));
    assert(As != NULL);

    int seed[] = {0, 0, 0, 1};
    lapack_int retval;
    retval = LAPACKE_zlarnv(1, seed, (size_t)lda*n, A);
    assert(retval == 0);

    plasma_complex32_t *AsRef = NULL;
    if (test) {
        AsRef = (plasma_complex32_t*)malloc(
            (size_t)ldas*n*sizeof(plasma_complex32_t));
        assert(AsRef != NULL);

        memcpy(AsRef, As, (size_t)ldas*n*sizeof(plasma_complex32_t));
    }

    //================================================================
    // Run and time PLASMA
    //================================================================
    plasma_time_t start = omp_get_wtime();

    retval = plasma_zlag2c(m, n, A, lda, As, ldas);

    plasma_time_t stop = omp_get_wtime();

    param[PARAM_GFLOPS].d  = 0.0;

    if (retval != PlasmaSuccess) {
        plasma_error("plasma_zlag2c() failed");
        param[PARAM_TIME].d    = 0.0;
        param[PARAM_ERROR].d   = 1.0;
        param[PARAM_SUCCESS].i = false;
        return;
    }
    else {
        param[PARAM_TIME].d = stop-start;
    }

    //================================================================
    // Test results by comparing to result of LAPACK function
    //================================================================
    if (test) {
        // Calculate relative error |As_ref - As|_F / |As_ref|_F < 3*eps
        // Using 3*eps covers complex arithmetic

        lapack_int mtrxLayout = LAPACK_COL_MAJOR;

        retval = LAPACKE_zlag2c_work(mtrxLayout, m, n, A, lda, AsRef, ldas);

        if (retval != PlasmaSuccess) {
            plasma_coreblas_error("LAPACKE_zlag2c_work() failed");
            param[PARAM_ERROR].d   = 1.0;
            param[PARAM_SUCCESS].i = false;
            return;
        }

        float work[1];

        // Calculate Frobenius norm of reference result As_ref
        double AsNormRef = LAPACKE_clange_work(mtrxLayout, 'F',
                                               ldas, n, AsRef, ldas, work);

        // Calculate difference As_ref-As
        plasma_complex32_t cmone = -1.0;
        cblas_caxpy((size_t)ldas*n, CBLAS_SADDR(cmone), As, 1, AsRef, 1);

        // Calculate Frobenius norm of As_ref-As
        double AsNormDiff = LAPACKE_clange_work(mtrxLayout, 'F',
                                                ldas, n, AsRef, ldas, work);

        // Calculate relative error |As_ref-As|_F / |As_ref|_F
        double error = AsNormDiff/AsNormRef;

        param[PARAM_ERROR].d   = error;
        param[PARAM_SUCCESS].i = error < 3*eps;
    }

    //================================================================
    // Free arrays
    //================================================================
    free(A);
    free(As);

    if (test)
        free(AsRef);
}
