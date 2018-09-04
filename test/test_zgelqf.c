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
 * @brief Tests ZGELQF.
 *
 * @param[in,out] param - array of parameters
 * @param[in]     run - whether to run test
 *
 * Sets flags in param indicating which parameters are used.
 * If run is true, also runs test and stores output parameters.
 ******************************************************************************/
void test_zgelqf(param_value_t param[], bool run)
{
    //================================================================
    // Mark which parameters are used.
    //================================================================
    param[PARAM_DIM    ].used = PARAM_USE_M | PARAM_USE_N;
    param[PARAM_PADA   ].used = true;
    param[PARAM_NB     ].used = true;
    param[PARAM_IB     ].used = true;
    param[PARAM_HMODE  ].used = true;
    if (! run)
        return;

    //================================================================
    // Set parameters.
    //================================================================
    int m = param[PARAM_DIM].dim.m;
    int n = param[PARAM_DIM].dim.n;

    int lda = imax(1, m + param[PARAM_PADA].i);

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

    int seed[] = {0, 0, 0, 1};
    lapack_int retval;
    retval = LAPACKE_zlarnv(1, seed, (size_t)lda*n, A);
    assert(retval == 0);

    plasma_complex64_t *Aref = NULL;
    if (test) {
        Aref = (plasma_complex64_t*)malloc(
            (size_t)lda*n*sizeof(plasma_complex64_t));
        assert(Aref != NULL);

        memcpy(Aref, A, (size_t)lda*n*sizeof(plasma_complex64_t));
    }

    //================================================================
    // Prepare the descriptor for matrix T.
    //================================================================
    plasma_desc_t T;

    //================================================================
    // Run and time PLASMA.
    //================================================================
    plasma_time_t start = omp_get_wtime();
    plasma_zgelqf(m, n, A, lda, &T);
    plasma_time_t stop = omp_get_wtime();
    plasma_time_t time = stop-start;

    param[PARAM_TIME].d = time;
    param[PARAM_GFLOPS].d = flops_zgelqf(m, n) / time / 1e9;

    //=================================================================
    // Test results by checking orthogonality of Q and precision of L*Q
    //=================================================================
    if (test) {
        // Check the orthogonality of Q.
        int minmn = imin(m, n);

        // Allocate space for Q.
        int ldq = minmn;
        plasma_complex64_t *Q =
            (plasma_complex64_t *)malloc((size_t)ldq*n*
                                         sizeof(plasma_complex64_t));
        assert(Q != NULL);

        // Build Q.
        plasma_zunglq(minmn, n, minmn, A, lda, T, Q, ldq);

        // Build the identity matrix
        plasma_complex64_t *Id =
            (plasma_complex64_t *) malloc((size_t)minmn*minmn*
                                          sizeof(plasma_complex64_t));
        assert(Id != NULL);
        LAPACKE_zlaset_work(LAPACK_COL_MAJOR, 'g', minmn, minmn,
                            0.0, 1.0, Id, minmn);

        // Perform Id - Q * Q^H
        cblas_zherk(CblasColMajor, CblasUpper, CblasNoTrans, minmn, n,
                    -1.0, Q, ldq, 1.0, Id, minmn);

        // work array of size m is needed for computing L_oo norm
        double *work = (double *) malloc((size_t)m*sizeof(double));
        assert(work != NULL);

        // |Id - Q * Q^H|_oo
        double ortho = LAPACKE_zlanhe_work(LAPACK_COL_MAJOR, 'I', 'u',
                                           minmn, Id, minmn, work);

        // normalize the result
        // |Id - Q * Q^H|_oo / n
        ortho /= minmn;

        free(Q);
        free(Id);

        // Check the accuracy of A - L * Q
        // LAPACK version does not construct Q, it uses only application of it

        // Extract the L.
        plasma_complex64_t *L =
            (plasma_complex64_t *)malloc((size_t)m*n*
                                         sizeof(plasma_complex64_t));
        assert(L != NULL);
        LAPACKE_zlaset_work(LAPACK_COL_MAJOR, 'u', m, n,
                            0.0, 0.0, L, m);
        LAPACKE_zlacpy_work(LAPACK_COL_MAJOR, 'l', m, n, A, lda, L, m);

        // Compute L * Q.
        plasma_zunmlq(PlasmaRight, PlasmaNoTrans, m, n, minmn, A, lda, T,
                      L, m);

        // Compute the difference.
        // L = A - L*Q
        for (int j = 0; j < n; j++)
            for (int i = 0; i < m; i++)
                L[j*m+i] = Aref[j*lda+i] - L[j*m+i];

        // |A|_oo
        double normA = LAPACKE_zlange_work(LAPACK_COL_MAJOR, 'I', m, n,
                                           Aref, lda, work);

        // |A - L*Q|_oo
        double error = LAPACKE_zlange_work(LAPACK_COL_MAJOR, 'I', m, n,
                                           L, m, work);

        // normalize the result
        // |A-LQ|_oo / (|A|_oo * n)
        error /= (normA * n);

        param[PARAM_ERROR].d = error;
        param[PARAM_ORTHO].d = ortho;
        param[PARAM_SUCCESS].i = (error < tol && ortho < tol);

        free(work);
        free(L);
    }

    //================================================================
    // Free arrays.
    //================================================================
    plasma_desc_destroy(&T);
    free(A);
    if (test)
        free(Aref);
}
