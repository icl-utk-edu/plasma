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
#include "plasma_core_blas.h"
#include "core_lapack.h"
#include "plasma.h"

#include <assert.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#undef  REAL
#define COMPLEX

/***************************************************************************//**
 *
 * @brief Tests ZHEEVD.
 *
 * @param[in]  param - array of parameters
 * @param[out] info  - string of column labels or column values; length InfoLen
 *
 * If param is NULL and info is NULL,     print usage and return.
 * If param is NULL and info is non-NULL, set info to column labels and return.
 * If param is non-NULL and info is non-NULL, set info to column values
 * and run test.
 ******************************************************************************/
void test_zheevd(param_value_t param[], bool run)
{
    //================================================================
    // Mark which parameters are used.
    //================================================================
    param[PARAM_JOB   ].used = true;
    param[PARAM_UPLO  ].used = true;
    param[PARAM_DIM   ].used = PARAM_USE_N;
    param[PARAM_PADA  ].used = true;
    param[PARAM_NB    ].used = true;
    param[PARAM_IB    ].used = true;
    param[PARAM_HMODE ].used = true;
    if (! run)
        return;

    //================================================================
    // Set parameters.
    //================================================================
    plasma_enum_t job  = plasma_job_const(param[PARAM_JOB].c);
    plasma_enum_t uplo = plasma_uplo_const(param[PARAM_UPLO].c);

    int n = param[PARAM_DIM].dim.n;

    int lda = imax(1, n + param[PARAM_PADA].i);

    int    test = param[PARAM_TEST].c == 'y';
    double tol  = param[PARAM_TOL].d * LAPACKE_dlamch('E');

    //================================================================
    // Set tuning parameters.
    //================================================================
    plasma_set(PlasmaNb, param[PARAM_NB].i);
    plasma_set(PlasmaIb, param[PARAM_NB].i/4);

    //================================================================
    // Allocate and initialize arrays.
    //================================================================
    plasma_complex64_t *A = (plasma_complex64_t *)malloc(
        (size_t)n*lda*sizeof(plasma_complex64_t));

    plasma_complex64_t *Aref = NULL;
    plasma_complex64_t *Q    = NULL;
    double             *Lambda_ref = NULL;
    plasma_complex64_t *work = NULL;
    double             *Lambda = (double*)malloc((size_t)n*sizeof(double));
    int seed[] = {0, 0, 0, 1};
    if (test) {
        Lambda_ref = (double*)malloc((size_t)n*sizeof(double));
        work = (plasma_complex64_t *)malloc(
            (size_t)3*n*sizeof(plasma_complex64_t));

        for (int i = 0; i < n; ++i) {
            Lambda_ref[i] = (double)i + 1;
        }

        int    mode  = 0;
        double dmax  = 1.0;
        double rcond = 1.0e6;
        LAPACKE_zlatms_work(LAPACK_COL_MAJOR, n, n,
                           'S', seed,
                           'H', Lambda_ref, mode, rcond,
                            dmax, n, n,
                           'N', A, lda, work);

        // Sort the eigenvalues
        LAPACKE_dlasrt_work( 'I', n, Lambda_ref );

        // Copy A into Aref
        Aref = (plasma_complex64_t *)malloc(
            (size_t)n*lda*sizeof(plasma_complex64_t));
        LAPACKE_zlacpy_work(LAPACK_COL_MAJOR,
                            'A', n, n, A, lda, Aref, lda);
    }
    else {
        LAPACKE_zlarnv(1, seed, (size_t)lda*n, A);
    }


    int ldq = lda;
    if (job == PlasmaNoVec) {
        Q = (plasma_complex64_t *)malloc(
            (size_t)n*ldq*sizeof(plasma_complex64_t));
    }

    //================================================================
    // Prepare the descriptor for matrix T.
    //================================================================
    plasma_desc_t T;

    //================================================================
    // Run and time PLASMA.
    //================================================================
    plasma_time_t start = omp_get_wtime();

    plasma_zheevd(job, uplo, n, A, lda, &T, Lambda, Q, ldq);
    //LAPACKE_zheevd( LAPACK_COL_MAJOR,
    //               'N', 'L',  n, A, lda, Lambda);
    plasma_time_t stop = omp_get_wtime();
    plasma_time_t time = stop - start;

    param[PARAM_TIME].d = time;
    param[PARAM_GFLOPS].d = flops_zgeqrf(n, n) / time / 1e9;

    if (test) {
        // Check the correctness of the eigenvalues values.
        double error = 0;
        for (int i = 0; i < n; ++i) {
            error += fabs( Lambda[i] - Lambda_ref[i] )
                     / fabs( Lambda_ref[i] );
        }

        error /= n*40;
        // Othorgonality test
        double done  =  1.0;
        double mdone = -1.0;

        // Build the idendity matrix
        plasma_complex64_t *Id
            = (plasma_complex64_t *) malloc(n*n*sizeof(plasma_complex64_t));
        LAPACKE_zlaset_work(LAPACK_COL_MAJOR, 'A', n, n, 0., 1., Id, n);

        double ortho = 0.;
        if (job == PlasmaVec) {
            // Perform Id - Q^H Q
            cblas_zherk(
                CblasColMajor, CblasUpper, CblasConjTrans,
                n, n, done, Q, n, mdone, Id, n);
            double normQ = LAPACKE_zlanhe_work(
                LAPACK_COL_MAJOR, 'I', 'U', n, Id, n, (double*)work);
            ortho = normQ/n;
        }
        param[PARAM_ERROR].d = error;
        param[PARAM_ORTHO].d = ortho;
        param[PARAM_SUCCESS].i = (error < tol && ortho < tol);

    }
    //================================================================
    // Free arrays.
    //================================================================
    // plasma_desc_destroy(&T);
    free(A);
    free(Q);
    free(Lambda);
    free(work);
    if (test) {
        free(Aref);
        free(Lambda_ref);
    }
}
