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
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <omp.h>

#define COMPLEX

/***************************************************************************//**
 *
 * @brief Tests DZAMAX.
 *
 * @param[in,out] param - array of parameters
 * @param[in]     run - whether to run test
 *
 * Sets flags in param indicating which parameters are used.
 * If run is true, also runs test and stores output parameters.
 ******************************************************************************/
void test_dzamax(param_value_t param[], bool run)
{
    //================================================================
    // Mark which parameters are used.
    //================================================================
    param[PARAM_COLROW ].used = true;
    param[PARAM_DIM    ].used = PARAM_USE_M | PARAM_USE_N;
    param[PARAM_PADA   ].used = true;
    param[PARAM_NB     ].used = true;
    if (! run)
        return;

    //================================================================
    // Set parameters
    //================================================================
    plasma_enum_t colrow = plasma_storev_const(param[PARAM_COLROW].c);

    int m = param[PARAM_DIM].dim.m;
    int n = param[PARAM_DIM].dim.n;

    int lda = imax(1, m + param[PARAM_PADA].i);

    int test = param[PARAM_TEST].c == 'y';

    //================================================================
    // Set tuning parameters.
    //================================================================
    plasma_set(PlasmaNb, param[PARAM_NB].i);

    //================================================================
    // Allocate and initialize arrays.
    //================================================================
    plasma_complex64_t *A =
        (plasma_complex64_t*)malloc((size_t)lda*n*sizeof(plasma_complex64_t));
    assert(A != NULL);

    int seed[] = {0, 0, 0, 1};
    lapack_int retval;
    retval = LAPACKE_zlarnv(1, seed, (size_t)lda*n, A);
    assert(retval == 0);

    size_t size = colrow == PlasmaColumnwise ? n : m;
    double *values = (double*)malloc(size*sizeof(double));
    assert(values != NULL);

    double *valref = NULL;
    if (test) {
        valref = (double*)malloc(size*sizeof(double));
        assert(valref != NULL);
    }

    //================================================================
    // Run and time PLASMA.
    //================================================================
    plasma_time_t start = omp_get_wtime();
    plasma_dzamax(colrow, m, n, A, lda, values);
    plasma_time_t stop = omp_get_wtime();
    plasma_time_t time = stop-start;

    // flops count each comparison as an addition
    plasma_enum_t norm =
        (colrow == PlasmaColumnwise ? PlasmaOneNorm : PlasmaInfNorm);
    param[PARAM_TIME].d = time;
    param[PARAM_GFLOPS].d = flops_zlange(m, n, norm) / time / 1e9;

    //================================================================
    // Test results by comparing to a reference implementation.
    //================================================================
    if (test) {
        if (colrow == PlasmaColumnwise) {
            for (int j = 0; j < n; j++) {
                CBLAS_INDEX idx = cblas_izamax(m, &A[lda*j], 1);
                valref[j] = plasma_core_dcabs1(A[lda*j+idx]);
            }
        }
        else {
            for (int i = 0; i < m; i++) {
                CBLAS_INDEX idx = cblas_izamax(n, &A[i], lda);
                valref[i] = plasma_core_dcabs1(A[i+lda*idx]);
            }
        }

        // Calculate difference.
        cblas_daxpy(size, -1.0, values, 1, valref, 1);

        // Set error to maximum difference.
        double error = valref[cblas_idamax(size, valref, 1)];

        param[PARAM_ERROR].d   = error;
        param[PARAM_SUCCESS].i = error == 0.0;
    }

    //================================================================
    // Free arrays.
    //================================================================
    free(A);
    free(values);
    if (test)
        free(valref);
}
