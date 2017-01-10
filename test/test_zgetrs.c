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
 * @brief Tests ZGETRS.
 *
 * @param[in]  param - array of parameters
 * @param[out] info  - string of column labels or column values; length InfoLen
 *
 * If param is NULL and info is NULL,     print usage and return.
 * If param is NULL and info is non-NULL, set info to column labels and return.
 * If param is non-NULL and info is non-NULL, set info to column values
 * and run test.
 ******************************************************************************/
void test_zgetrs(param_value_t param[], char *info)
{
    //================================================================
    // Print usage info or return column labels or values.
    //================================================================
    if (param == NULL) {
        if (info == NULL) {
            // Print usage info.
            print_usage(PARAM_N);
            print_usage(PARAM_NRHS);
            print_usage(PARAM_PADA);
            print_usage(PARAM_PADB);
            print_usage(PARAM_NB);
            print_usage(PARAM_IB);
            print_usage(PARAM_NTPF);
        }
        else {
            // Return column labels.
            snprintf(info, InfoLen,
                     "%*s %*s %*s %*s %*s %*s %*s",
                     InfoSpacing, "N",
                     InfoSpacing, "Nrhs",
                     InfoSpacing, "PadA",
                     InfoSpacing, "PadB",
                     InfoSpacing, "NB",
                     InfoSpacing, "IB",
                     InfoSpacing, "NTPF");
        }
        return;
    }
    // Return column values.
    snprintf(info, InfoLen,
             "%*d %*d %*d %*d %*d %*d %*d",
             InfoSpacing, param[PARAM_N].i,
             InfoSpacing, param[PARAM_NRHS].i,
             InfoSpacing, param[PARAM_PADA].i,
             InfoSpacing, param[PARAM_PADB].i,
             InfoSpacing, param[PARAM_NB].i,
             InfoSpacing, param[PARAM_IB].i,
             InfoSpacing, param[PARAM_NTPF].i);

    //================================================================
    // Set parameters.
    //================================================================
    int n = param[PARAM_N].i;
    int nrhs = param[PARAM_NRHS].i;

    int lda = imax(1, n+param[PARAM_PADA].i);
    int ldb = imax(1, n+param[PARAM_PADB].i);

    int test = param[PARAM_TEST].c == 'y';
    double tol = param[PARAM_TOL].d * LAPACKE_dlamch('E');

    //================================================================
    // Set tuning parameters.
    //================================================================
    plasma_set(PlasmaNb, param[PARAM_NB].i);
    plasma_set(PlasmaIb, param[PARAM_IB].i);
    plasma_set(PlasmaNumPanelThreads, param[PARAM_NTPF].i);

    //================================================================
    // Allocate and initialize arrays.
    //================================================================
    plasma_complex64_t *A =
        (plasma_complex64_t*)malloc((size_t)lda*n*sizeof(plasma_complex64_t));
    assert(A != NULL);

    plasma_complex64_t *B =
        (plasma_complex64_t*)malloc(
            (size_t)ldb*nrhs*sizeof(plasma_complex64_t));
    assert(B != NULL);

    int *ipiv = (int*)malloc((size_t)n*sizeof(int));
    assert(ipiv != NULL);

    int seed[] = {0, 0, 0, 1};
    lapack_int retval;
    retval = LAPACKE_zlarnv(1, seed, (size_t)lda*n, A);
    assert(retval == 0);

    retval = LAPACKE_zlarnv(1, seed, (size_t)ldb*nrhs, B);
    assert(retval == 0);

    plasma_complex64_t *Aref = NULL;
    plasma_complex64_t *Bref = NULL;
    double *work = NULL;
    if (test) {
        Aref = (plasma_complex64_t*)malloc(
            (size_t)lda*n*sizeof(plasma_complex64_t));
        assert(Aref != NULL);

        Bref = (plasma_complex64_t*)malloc(
            (size_t)ldb*nrhs*sizeof(plasma_complex64_t));
        assert(Bref != NULL);

        memcpy(Aref, A, (size_t)lda*n*sizeof(plasma_complex64_t));
        memcpy(Bref, B, (size_t)ldb*nrhs*sizeof(plasma_complex64_t));
    }

    //================================================================
    // Run GETRF
    //================================================================
    plasma_zgetrf(n, n, A, lda, ipiv);

    //================================================================
    // Run and time PLASMA.
    //================================================================
    plasma_time_t start = omp_get_wtime();
    plasma_zgetrs(n, nrhs, A, lda, ipiv, B, ldb);
    plasma_time_t stop = omp_get_wtime();
    plasma_time_t time = stop-start;

    param[PARAM_TIME].d = time;
    param[PARAM_GFLOPS].d = flops_zgetrs(n, nrhs) / time / 1e9;

    //================================================================
    // Test results by checking the residual
    //
    //                      || B - AX ||_I
    //                --------------------------- < epsilon
    //                 || A ||_I * || X ||_I * N
    //
    //================================================================
    if (test) {
        plasma_complex64_t zone  =  1.0;
        plasma_complex64_t zmone = -1.0;

        work = (double*)malloc((size_t)n*sizeof(double));
        assert(work != NULL);

        double Anorm = LAPACKE_zlange_work(
            LAPACK_COL_MAJOR, 'I', n, n, Aref, lda, work);
        double Xnorm = LAPACKE_zlange_work(
            LAPACK_COL_MAJOR, 'I', n, nrhs, B, ldb, work);

        // Bref -= Aref*B
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, nrhs, n,
                    CBLAS_SADDR(zmone), Aref, lda,
                                        B,    ldb,
                    CBLAS_SADDR(zone),  Bref, ldb);

        double Rnorm = LAPACKE_zlange_work(
            LAPACK_COL_MAJOR, 'I', n, nrhs, Bref, ldb, work);
        double residual = Rnorm/(n*Anorm*Xnorm);

        param[PARAM_ERROR].d = residual;
        param[PARAM_SUCCESS].i = residual < tol;
    }

    //================================================================
    // Free arrays.
    //================================================================
    free(A);
    free(B);
    free(ipiv);
    if (test) {
        free(Aref);
        free(Bref);
        free(work);
    }
}
