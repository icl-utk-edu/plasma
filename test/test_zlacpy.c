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

#include "core_blas.h"
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
 * @brief Tests ZLACPY
 *
 * @param[in]  param - array of parameters
 * @param[out] info  - string of column labels or column values; length InfoLen
 *
 * If param is NULL and info is NULL,     print usage and return.
 * If param is NULL and info is non-NULL, set info to column labels and return.
 * If param is non-NULL and info is non-NULL, set info to column values
 * and run test.
 ******************************************************************************/
void test_zlacpy(param_value_t param[], char *info)
{
    //================================================================
    // Print usage info or return column labels or values
    //================================================================
    if (param == NULL) {
        if (info == NULL) {
            // Print usage info
            print_usage(PARAM_UPLO);
<<<<<<< working copy
            print_usage(PARAM_TRANSA);
            print_usage(PARAM_M);
            print_usage(PARAM_N);
||||||| base
            print_usage(PARAM_M);
            print_usage(PARAM_N);
=======
            print_usage(PARAM_DIM);
>>>>>>> merge rev
            print_usage(PARAM_PADA);
            print_usage(PARAM_PADB);
            print_usage(PARAM_NB);
        }
        else {
            // Return column labels
            snprintf(info, InfoLen,
                     "%*s %*s %*s %*s %*s %*s %*s",
                     InfoSpacing, "UpLo",
                     InfoSpacing, "TransA",
                     InfoSpacing, "m",
                     InfoSpacing, "n",
                     InfoSpacing, "PadA",
                     InfoSpacing, "PadB",
                     InfoSpacing, "nb");
        }
        return;
    }
    // Return column values
    snprintf(info, InfoLen,
             "%*c %*c %*d %*d %*d %*d %*d",
             InfoSpacing, param[PARAM_UPLO].c,
<<<<<<< working copy
             InfoSpacing, param[PARAM_TRANSA].c,
             InfoSpacing, param[PARAM_M].i,
             InfoSpacing, param[PARAM_N].i,
||||||| base
             InfoSpacing, param[PARAM_M].i,
             InfoSpacing, param[PARAM_N].i,
=======
             InfoSpacing, param[PARAM_DIM].dim.m,
             InfoSpacing, param[PARAM_DIM].dim.n,
>>>>>>> merge rev
             InfoSpacing, param[PARAM_PADA].i,
             InfoSpacing, param[PARAM_PADB].i,
             InfoSpacing, param[PARAM_NB].i);

    //================================================================
    // Set parameters
    //================================================================
    plasma_enum_t uplo = plasma_uplo_const(param[PARAM_UPLO].c);
    plasma_enum_t transa = plasma_trans_const(param[PARAM_TRANSA].c);

    int m = param[PARAM_DIM].dim.m;
    int n = param[PARAM_DIM].dim.n;

    int lda = imax(1, m + param[PARAM_PADA].i);
    int ldb = imax(1, m + param[PARAM_PADB].i);

    int    test = param[PARAM_TEST].c == 'y';
    double eps  = LAPACKE_dlamch('E');

    //================================================================
    // Set tuning parameters
    //================================================================
    plasma_set(PlasmaNb, param[PARAM_NB].i);

    //================================================================
    // Allocate and initialize arrays
    //================================================================
    plasma_complex64_t *A =
        (plasma_complex64_t*)malloc((size_t)lda*n*sizeof(plasma_complex64_t));
    assert(A != NULL);

    plasma_complex64_t *B =
        (plasma_complex64_t*)malloc((size_t)ldb*n*sizeof(plasma_complex64_t));
    assert(B != NULL);

    int seed[] = {0, 0, 0, 1};
    lapack_int retval;
    retval = LAPACKE_zlarnv(1, seed, (size_t)lda*n, A);
    assert(retval == 0);

    lapack_int mtrxLayout = LAPACK_COL_MAJOR;

    // LAPACKE_[zcds]lantr_work has a bug (returns 0)
    // in MKL <= 11.3.3 (at least). Fixed in LAPACK 3.6.1.
    // For now, zero out the opposite triangle and use lange.
    // @sa test_ztrmm

    // Enforce zeroes in general rectangle or upper or lower triangle of A
    switch (uplo) {
        case PlasmaLower:
            LAPACKE_zlaset_work(
                mtrxLayout, 'U', m-1, n-1, 0.0, 0.0, &A[m], lda);
            break;
        case PlasmaUpper:
            LAPACKE_zlaset_work(
                mtrxLayout, 'L', m-1, n-1, 0.0, 0.0, &A[1], lda);
            break;
    }
    // Zero out B
    int Bm = (transa == PlasmaGeneral ? m : n);
    int Bn = (transa == PlasmaGeneral ? n : m);
    LAPACKE_zlaset_work(
        mtrxLayout, 'G', Bm, Bn, 0.0, 0.0, B, ldb);

    //================================================================
    // Run and time PLASMA
    //================================================================
    plasma_time_t start = omp_get_wtime();

    retval = plasma_zlacpy(uplo, transa, m, n, A, lda, B, ldb);

    plasma_time_t stop = omp_get_wtime();

    param[PARAM_GFLOPS].d = 0.0;

    if (retval != PlasmaSuccess) {
        plasma_error("plasma_zlacpy() failed");
        param[PARAM_TIME].d    = 0.0;
        param[PARAM_ERROR].d   = 1.0;
        param[PARAM_SUCCESS].i = false;
        return;
    }
    else {
        param[PARAM_TIME].d = stop-start;
    }

    //================================================================
    // Test results by comparing to result of core_zlacpy function
    //================================================================
    if (test) {
        // Calculate relative error |op(A) - B|_F / |A|_F < 3*eps
        // Using 3*eps covers complex arithmetic

        // Calculate difference op(A)-B
        if (transa == PlasmaTrans) {
            for (int i=0; i<m; i++)
                for (int j=0; j<n; j++)
                    B[j + i*ldb] -= A[i + j*lda];
        } else if (transa == PlasmaConjTrans) {
            for (int i=0; i<m; i++)
                for (int j=0; j<n; j++)
                    B[j + i*ldb] -= conj(A[i + j*lda]);
        } else {
            for (int i=0; i<m; i++)
                for (int j=0; j<n; j++)
                    B[i + j*ldb] -= A[i + j*lda];
        }
        if (retval != PlasmaSuccess) {
            coreblas_error("LAPACKE_zlacpy_work() failed");
            param[PARAM_ERROR].d   = 1.0;
            param[PARAM_SUCCESS].i = false;
            return;
        }

        double work[1];

        // Calculate Frobenius norm of A
        double Anorm = LAPACKE_zlange_work(
                           mtrxLayout, 'F', m, n, A, lda, work);


        // Calculate Frobenius norm of op(A)-B
        double BnormDiff = LAPACKE_zlange_work(
                               mtrxLayout, 'F', Bm, Bn, B, ldb, work);

        // Calculate relative error |op(A)-B|_F / |A|_F
        double error = BnormDiff/Anorm;

        param[PARAM_ERROR].d   = error;
        param[PARAM_SUCCESS].i = error < 3*eps;
    }

    //================================================================
    // Free arrays
    //================================================================
    free(A);
    free(B);
}
