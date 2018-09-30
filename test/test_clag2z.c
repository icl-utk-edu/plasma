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

#include <plasma_core_blas.h>
#include "core_lapack.h"
#include "flops.h"
#include "plasma.h"
#include "test.h"

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
 * @brief Tests CLAG2Z
 *
 * @param[in,out] param - array of parameters
 * @param[in]     run - whether to run test
 *
 * Sets flags in param indicating which parameters are used.
 * If run is true, also runs test and stores output parameters.
 ******************************************************************************/
void test_clag2z(param_value_t param[], bool run)
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

    int ldas = imax(1, m + param[PARAM_PADA].i);
    int lda  = imax(1, m + param[PARAM_PADB].i);

    int    test = param[PARAM_TEST].c == 'y';
    double eps  = LAPACKE_dlamch('E');

    //================================================================
    // Set tuning parameters
    //================================================================
    plasma_set(PlasmaNb, param[PARAM_NB].i);

    //================================================================
    // Allocate and initialize arrays
    //================================================================
    plasma_complex32_t *As =
        (plasma_complex32_t*)malloc((size_t)ldas*n*sizeof(plasma_complex32_t));
    assert(As != NULL);

    plasma_complex64_t *A =
        (plasma_complex64_t*)malloc((size_t)lda*n*sizeof(plasma_complex64_t));
    assert(A != NULL);

    int seed[] = {0, 0, 0, 1};
    lapack_int retval;
    retval = LAPACKE_clarnv(1, seed, (size_t)ldas*n, As);
    assert(retval == 0);

    plasma_complex64_t *Aref = NULL;
    if (test) {
        Aref = (plasma_complex64_t*)malloc(
            (size_t)ldas*n*sizeof(plasma_complex64_t));
        assert(Aref != NULL);

        memcpy(Aref, A, (size_t)lda*n*sizeof(plasma_complex64_t));
    }

    //================================================================
    // Run and time PLASMA
    //================================================================
    plasma_time_t start = omp_get_wtime();

    retval = plasma_clag2z(m, n, As, ldas, A, lda);

    plasma_time_t stop = omp_get_wtime();

    param[PARAM_GFLOPS].d = 0.0;

    if (retval != PlasmaSuccess) {
        plasma_error("plasma_clag2z() failed");
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
        // Calculate relative error |A_ref - A|_F / |A_ref|_F < 3*eps
        // Using 3*eps covers complex arithmetic

        lapack_int mtrxLayout = LAPACK_COL_MAJOR;

        retval = LAPACKE_clag2z_work(mtrxLayout, m, n, As, ldas, Aref, lda);

        if (retval != PlasmaSuccess) {
            plasma_coreblas_error("LAPACKE_clag2z_work() failed");
            param[PARAM_ERROR].d   = 1.0;
            param[PARAM_SUCCESS].i = false;
            return;
        }

        double work[1];

        // Calculate Frobenius norm of reference result A_ref
        double AnormRef = LAPACKE_zlange_work(mtrxLayout, 'F',
                                              lda, n, Aref, lda, work);

        // Calculate difference A_ref-A
        plasma_complex64_t zmone = -1.0;
        cblas_zaxpy((size_t)lda*n, CBLAS_SADDR(zmone), A, 1, Aref, 1);

        // Calculate Frobenius norm of A_ref-A
        double AnormDiff = LAPACKE_zlange_work(mtrxLayout, 'F',
                                               lda, n, Aref, lda, work);

        // Calculate relative error |A_ref-A|_F / |A_ref|_F
        double error = AnormDiff/AnormRef;

        param[PARAM_ERROR].d   = error;
        param[PARAM_SUCCESS].i = error < 3*eps;
    }

    //================================================================
    // Free arrays
    //================================================================
    free(As);
    free(A);

    if (test)
        free(Aref);
}
