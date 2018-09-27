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

#include "plasma.h"
#include <plasma_core_blas.h>
#include "core_lapack.h"
#include "flops.h"
#include "test.h"

#include <assert.h>
#include <math.h>
#include <omp.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define COMPLEX

#define A(i_, j_) A[(i_) + (size_t)lda*(j_)]

/***************************************************************************//**
 *
 * @brief Tests ZCPOSV
 *
 * @param[in,out] param - array of parameters
 * @param[in]     run - whether to run test
 *
 * Sets flags in param indicating which parameters are used.
 * If run is true, also runs test and stores output parameters.
 ******************************************************************************/
void test_zcposv(param_value_t param[], bool run)
{
    //================================================================
    // Mark which parameters are used.
    //================================================================
    param[PARAM_UPLO   ].used = true;
    param[PARAM_DIM    ].used = PARAM_USE_N;
    param[PARAM_NRHS   ].used = true;
    param[PARAM_PADA   ].used = true;
    param[PARAM_PADB   ].used = true;
    param[PARAM_NB     ].used = true;
    param[PARAM_ZEROCOL].used = true;
    param[PARAM_ITERSV ].used = true;
    if (! run)
        return;

    //================================================================
    // Set parameters
    //================================================================
    plasma_enum_t uplo = plasma_uplo_const(param[PARAM_UPLO].c);

    int n    = param[PARAM_DIM].dim.n;
    int nrhs = param[PARAM_NRHS].i;
    int lda  = imax(1, n + param[PARAM_PADA].i);
    int ldb  = imax(1, n + param[PARAM_PADB].i);
    int ldx  = ldb;
    int ITER;

    int    test = param[PARAM_TEST].c == 'y';
    double tol  = param[PARAM_TOL].d * LAPACKE_dlamch('E');

    //================================================================
    // Set tuning parameters
    //================================================================
    plasma_set(PlasmaTuning, PlasmaDisabled);
    plasma_set(PlasmaNb, param[PARAM_NB].i);

    //================================================================
    // Allocate and initialize arrays
    //================================================================
    plasma_complex64_t *A = (plasma_complex64_t *)malloc(
    (size_t)lda*n*sizeof(plasma_complex64_t));
    assert(A != NULL);

    plasma_complex64_t *B = (plasma_complex64_t *)malloc(
        (size_t)ldb*nrhs*sizeof(plasma_complex64_t));
    assert(B != NULL);

    plasma_complex64_t *X = (plasma_complex64_t *)malloc(
    (size_t)ldx*nrhs*sizeof(plasma_complex64_t));
    assert(X != NULL);

    // Initialize A for random Hermitian (Symmetric) matrix
    int seed[] = {0, 0, 0, 1};
    lapack_int retval;
    retval = LAPACKE_zlarnv(1, seed, (size_t)lda*n, A);
    assert(retval == 0);

    for (int i = 0; i < n; ++i) {
        A(i,i) = creal(A(i,i)) + n;
        for (int j = 0; j < i; ++j) {
            A(j,i) = conj(A(i,j));
        }
    }

    int zerocol = param[PARAM_ZEROCOL].i;
    if (zerocol >= 0 && zerocol < n) {
        LAPACKE_zlaset_work(
            LAPACK_COL_MAJOR, 'F', n, 1, 0.0, 0.0, &A(0, zerocol), lda);
        LAPACKE_zlaset_work(
            LAPACK_COL_MAJOR, 'F', 1, n, 0.0, 0.0, &A(zerocol, 0), lda);
    }

    // Initialize B
    retval = LAPACKE_zlarnv(1, seed, (size_t)ldb*nrhs, B);
    assert(retval == 0);

    plasma_complex64_t *Aref = NULL;
    if (test) {
        Aref = (plasma_complex64_t *)malloc(
            (size_t)lda*n*sizeof(plasma_complex64_t));
        assert(Aref != NULL);
        memcpy(Aref, A, (size_t)lda*n*sizeof(plasma_complex64_t));
    }

    //================================================================
    // Run and time PLASMA
    //================================================================
    plasma_time_t start = omp_get_wtime();
    int plainfo = plasma_zcposv(uplo, n, nrhs, A, lda, B, ldb, X, ldx, &ITER);
    plasma_time_t stop = omp_get_wtime();
    plasma_time_t time = stop-start;
    double flops = flops_zpotrf(n) + flops_zpotrs(n, nrhs);
    param[PARAM_ITERSV].i = ITER;
    param[PARAM_TIME].d   = time;
    param[PARAM_GFLOPS].d = flops / time / 1e9;

    //================================================================
    // Test results by checking the residual
    //
    //                      || B - AX ||_I
    //                --------------------------- < epsilon
    //                 || A ||_I * || X ||_I * N
    //
    //================================================================
    if (test) {
        if (plainfo == 0) {
            plasma_complex64_t alpha =  1.0;
            plasma_complex64_t beta  = -1.0;

            lapack_int mtrxLayout = LAPACK_COL_MAJOR;
            lapack_int mtrxNorm   = 'I';

            double *work = (double *)malloc(n*sizeof(double));
            assert(work != NULL);

            // Calculate infinite norms of matrices A_ref and X
            double Anorm = LAPACKE_zlange_work(mtrxLayout, mtrxNorm, n, n, Aref,
                                               lda, work);
            double Xnorm = LAPACKE_zlange_work(mtrxLayout, mtrxNorm, n, nrhs, X,
                                               ldx, work);

            // Calculate residual R = A*X-B, store result in B
            cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, nrhs, n,
                        CBLAS_SADDR(alpha), Aref, lda,
                                            X,    ldx,
                        CBLAS_SADDR(beta),  B,    ldb);

            // Calculate infinite norm of residual matrix R
            double Rnorm = LAPACKE_zlange_work(mtrxLayout, mtrxNorm, n, nrhs, B,
                                               ldb, work);
            // Calculate relative error
            double residual = Rnorm / ( n*Anorm*Xnorm );

            param[PARAM_ERROR].d   = residual;
            param[PARAM_SUCCESS].i = residual < tol;

            free(work);
        }
        else {
            int lapinfo = LAPACKE_zcposv(
                              LAPACK_COL_MAJOR, lapack_const(uplo),
                              n, nrhs, A, lda, B, ldb, X, ldx, &ITER);
            if (plainfo == lapinfo) {
                param[PARAM_ERROR].d = 0.0;
                param[PARAM_SUCCESS].i = 1;
            }
            else {
                param[PARAM_ERROR].d = INFINITY;
                param[PARAM_SUCCESS].i = 0;
            }
        }
        free(Aref);
    }

    //================================================================
    // Free arrays
    //================================================================
    free(A); free(B); free(X);
}
