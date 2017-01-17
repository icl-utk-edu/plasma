/**
 *
 * @file
 *
 *  PLASMA is a software package provided by:
 *  University of Tennessee, US,
 *  University of Manchester, UK.
 *
 * @precisions normal z -> c d s
 *
 **/
#include "test.h"
#include "flops.h"
#include "core_blas.h"
#include "core_lapack.h"
#include "plasma.h"

#include <assert.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <omp.h>

#define COMPLEX

#define A(i_, j_) A[(i_) + (size_t)lda*(j_)]

/***************************************************************************//**
 *
 * @brief Tests ZTRSM.
 *
 * @param[in]  param - array of parameters
 * @param[out] info  - string of column labels or column values; length InfoLen
 *
 * If param is NULL and info is NULL,     print usage and return.
 * If param is NULL and info is non-NULL, set info to column labels and return.
 * If param is non-NULL and info is non-NULL, set info to column values
 * and run test.
 ******************************************************************************/
void test_zgetri_aux(param_value_t param[], char *info)
{
    //================================================================
    // Print usage info or return column labels or values.
    //================================================================
    if (param == NULL) {
        if (info == NULL) {
            // Print usage info.
            print_usage(PARAM_N);
            print_usage(PARAM_PADA);
            print_usage(PARAM_NB);
        }
        else {
            // Return column labels.
            snprintf(info, InfoLen,
                     "%*s %*s %*s",
                     InfoSpacing, "N",
                     InfoSpacing, "PadA",
                     InfoSpacing, "NB");
        }
        return;
    }
    // Return column values.
    snprintf(info, InfoLen,
             "%*d %*d %*d",
             InfoSpacing, param[PARAM_N].i,
             InfoSpacing, param[PARAM_PADA].i,
             InfoSpacing, param[PARAM_NB].i);

    //================================================================
    // Set parameters.
    //================================================================
    int n = param[PARAM_N].i;
    int lda = imax(1, n + param[PARAM_PADA].i);

    int test = param[PARAM_TEST].c == 'y';
    double tol = param[PARAM_TOL].d * LAPACKE_dlamch('E');

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

    int *ipiv;
    ipiv = (int*)malloc((size_t)n*sizeof(int));
    assert(ipiv != NULL);

    int seed[] = {0, 0, 0, 1};
    lapack_int retval;

    //=================================================================
    // Initialize the matrices.
    // Factor A into LU to get well-conditioned triangular matrices.
    // Use L for unit triangle, and U for non-unit triangle,
    // transposing as necessary.
    // (There is some danger, as L^T or U^T may be much worse conditioned
    // than L or U, but in practice it seems okay.
    // See Higham, Accuracy and Stability of Numerical Algorithms, ch 8.)
    //=================================================================
    retval = LAPACKE_zlarnv(1, seed, (size_t)lda*n, A);
    assert(retval == 0);
    LAPACKE_zgetrf(CblasColMajor, n, n, A, lda, ipiv);
    plasma_complex64_t *L = NULL;
    plasma_complex64_t *U = NULL;
    if (test) {
        L = (plasma_complex64_t*)malloc((size_t)n*n*sizeof(plasma_complex64_t));
        assert(L != NULL);
        U = (plasma_complex64_t*)malloc((size_t)n*n*sizeof(plasma_complex64_t));
        assert(U != NULL);

        LAPACKE_zlacpy_work(LAPACK_COL_MAJOR, 'F', n, n, A, lda, L, n);
        LAPACKE_zlacpy_work(LAPACK_COL_MAJOR, 'F', n, n, A, lda, U, n);

        for (int j = 0; j < n; j++) {
            L[j + j*n] = 1.0;
            for (int i = 0; i < j; i++) L[i + j*n] = 0.0;
            for (int i = j+1; i < n; i++) U[i + j*n] = 0.0;
        }
    }

    //================================================================
    // Run and time PLASMA.
    //================================================================
    plasma_time_t start = omp_get_wtime();

    plasma_zgetri_aux( n, A, lda );

    plasma_time_t stop = omp_get_wtime();
    plasma_time_t time = stop-start;

    param[PARAM_TIME].d = time;
    param[PARAM_GFLOPS].d = flops_ztrsm(PlasmaRight, n, n) / time / 1e9;

    //================================================================
    // Test results by checking the residual
    // ||A*L - B|| / (||A||*||L||)
    //================================================================
    if (test) {
        plasma_complex64_t zone  =  1.0;
        plasma_complex64_t zmone = -1.0;
        double work[1];
        double Anorm = LAPACKE_zlange_work(
                           LAPACK_COL_MAJOR, 'F', n, n, A, lda, work);
        double Lnorm = LAPACKE_zlange_work(
                           LAPACK_COL_MAJOR, 'F', n, n, L, n, work);

        // A*L - U
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n,
                    CBLAS_SADDR(zone),  A, lda,
                                        L, n,
                    CBLAS_SADDR(zmone), U, n);

        double error = LAPACKE_zlange_work(
                           LAPACK_COL_MAJOR, 'F', n, n, U, n, work);
        param[PARAM_ERROR].d = error / (Anorm * Lnorm);
        param[PARAM_SUCCESS].i = error < tol;
    }

    //================================================================
    // Free arrays.
    //================================================================
    free(A);
    free(ipiv);
    if (test) {
        free(L); free(U);
    }
}
