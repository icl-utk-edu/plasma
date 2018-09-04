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
#include "core_lapack.h"

#include <assert.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <omp.h>

#define COMPLEX

/***************************************************************************//**
 *
 * @brief Tests ZGELQS.
 *
 * @param[in,out] param - array of parameters
 * @param[in]     run - whether to run test
 *
 * Sets flags in param indicating which parameters are used.
 * If run is true, also runs test and stores output parameters.
 ******************************************************************************/
void test_zgelqs(param_value_t param[], bool run)
{
    //================================================================
    // Mark which parameters are used.
    //================================================================
    param[PARAM_DIM    ].used = PARAM_USE_M | PARAM_USE_N;
    param[PARAM_NRHS   ].used = true;
    param[PARAM_PADA   ].used = true;
    param[PARAM_PADB   ].used = true;
    param[PARAM_NB     ].used = true;
    param[PARAM_IB     ].used = true;
    param[PARAM_HMODE  ].used = true;
    if (! run)
        return;

    //================================================================
    // Set parameters.
    //================================================================
    int m    = param[PARAM_DIM].dim.m;
    int n    = param[PARAM_DIM].dim.n;
    int nrhs = param[PARAM_NRHS].i;

    int lda = imax(1, m + param[PARAM_PADA].i);
    int ldb = imax(1, imax(m, n) + param[PARAM_PADB].i);

    int test = param[PARAM_TEST].c == 'y';
    double tol = param[PARAM_TOL].d * LAPACKE_dlamch('E');

    //================================================================
    // Set tuning parameters.
    //================================================================
    plasma_set(PlasmaTuning, PlasmaDisabled);
    plasma_set(PlasmaNb, param[PARAM_NB].i);
    plasma_set(PlasmaIb, param[PARAM_IB].i);
    if (param[PARAM_HMODE].c == 't')
        plasma_set(PlasmaHouseholderMode, PlasmaTreeHouseholder);
    else
        plasma_set(PlasmaHouseholderMode, PlasmaFlatHouseholder);

    //================================================================
    // Allocate and initialize arrays.
    //================================================================
    plasma_complex64_t *A =
        (plasma_complex64_t*)malloc((size_t)lda*n*sizeof(plasma_complex64_t));
    assert(A != NULL);

    plasma_complex64_t *B =
        (plasma_complex64_t*)malloc((size_t)ldb*nrhs*
                                    sizeof(plasma_complex64_t));
    assert(B != NULL);

    int seed[] = {0, 0, 0, 1};
    lapack_int retval;
    retval = LAPACKE_zlarnv(1, seed, (size_t)lda*n, A);
    assert(retval == 0);

    retval = LAPACKE_zlarnv(1, seed, (size_t)ldb*nrhs, B);
    assert(retval == 0);

    // store the original arrays if residual is to be evaluated
    plasma_complex64_t *Aref = NULL;
    plasma_complex64_t *Bref = NULL;
    if (test) {
        Aref = (plasma_complex64_t*)malloc(
            (size_t)lda*n*sizeof(plasma_complex64_t));
        assert(Aref != NULL);

        memcpy(Aref, A, (size_t)lda*n*sizeof(plasma_complex64_t));

        Bref = (plasma_complex64_t*)malloc(
            (size_t)ldb*nrhs*sizeof(plasma_complex64_t));
        assert(Bref != NULL);

        memcpy(Bref, B, (size_t)ldb*nrhs*sizeof(plasma_complex64_t));
    }

    //================================================================
    // Prepare the descriptor for matrix T.
    //================================================================
    plasma_desc_t T;

    //================================================================
    // Run and time PLASMA.
    //================================================================
    // prepare LQ factorization of A - only auxiliary for this test,
    // time is not measured
    plasma_zgelqf(m, n, A, lda, &T);

    // perform solution of the system by the prepared LQ factorization of A
    plasma_time_t start = omp_get_wtime();
    plasma_zgelqs(m, n, nrhs,
                  A, lda,
                  T,
                  B, ldb);

    plasma_time_t stop = omp_get_wtime();
    plasma_time_t time = stop-start;

    param[PARAM_TIME].d = time;
    param[PARAM_GFLOPS].d =
        (flops_ztrsm(PlasmaLeft, m, nrhs) +
         flops_zunmlq(PlasmaLeft, n, nrhs, m)) / time / 1e9;

    //================================================================
    // Test results by solving a linear system.
    //================================================================
    if (test) {
        // |A|_F
        double work[1];
        double Anorm = LAPACKE_zlange_work(LAPACK_COL_MAJOR, 'F', m, n,
                                           Aref, lda, work);

        // |B|_F
        double Bnorm = LAPACKE_zlange_work(LAPACK_COL_MAJOR, 'F', m, nrhs,
                                           Bref, ldb, work);

        // |X|_F, solution X is now stored in the n-by-nrhs part of B
        double Xnorm = LAPACKE_zlange_work(LAPACK_COL_MAJOR, 'F', n, nrhs,
                                           B, ldb, work);

        // compute residual and store it in B = A*X - B
        plasma_complex64_t zone  =  1.0;
        plasma_complex64_t zmone = -1.0;
        plasma_complex64_t zzero =  0.0;
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, nrhs, n,
                    CBLAS_SADDR(zone), Aref, lda, B, ldb,
                    CBLAS_SADDR(zmone), Bref, ldb);

        // Compute B = A^H * (A*X - B)
        cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, n, nrhs, m,
                    CBLAS_SADDR(zone), Aref, lda, Bref, ldb,
                    CBLAS_SADDR(zzero), B, ldb);

        // |RES|_F
        double Rnorm = LAPACKE_zlange_work(LAPACK_COL_MAJOR, 'F', n, nrhs,
                                           B, ldb, work);

        // normalize the result
        double result = Rnorm / ( (Anorm*Xnorm+Bnorm)*n);

        param[PARAM_ERROR].d = result;
        param[PARAM_SUCCESS].i = result < tol;
    }

    //================================================================
    // Free arrays.
    //================================================================
    plasma_desc_destroy(&T);
    free(A);
    free(B);
    if (test) {
        free(Aref);
        free(Bref);
    }
}
