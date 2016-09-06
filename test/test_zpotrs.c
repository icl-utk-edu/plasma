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
#include "core_lapack.h"

#include <assert.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <omp.h>
#include <plasma.h>

#define COMPLEX

#define A(i,j)  A[i + j*lda]

/***************************************************************************//**
 *
 * @brief Tests ZPOTRS.
 *
 * @param[in]  param - array of parameters
 * @param[out] info  - string of column labels or column values; length InfoLen
 *
 * If param is NULL and info is NULL,     print usage and return.
 * If param is NULL and info is non-NULL, set info to column labels and return.
 * If param is non-NULL and info is non-NULL, set info to column values
 * and run test.
 ******************************************************************************/
void test_zpotrs(param_value_t param[], char *info)
{
    //================================================================
    // Print usage info or return column labels or values.
    //================================================================
    if (param == NULL) {
        if (info == NULL) {
            // Print usage info.
            print_usage(PARAM_UPLO);
            print_usage(PARAM_N);
            print_usage(PARAM_NRHS);
            print_usage(PARAM_PADA);
            print_usage(PARAM_PADB);
            print_usage(PARAM_NB);
        }
        else {
            // Return column labels.
            snprintf(info, InfoLen,
                     "%*s %*s %*s %*s %*s %*s",
                     InfoSpacing, "Uplo",
                     InfoSpacing, "N",
                     InfoSpacing, "NRHS",
                     InfoSpacing, "PadA",
                     InfoSpacing, "PadB",
                     InfoSpacing, "NB");
        }
        return;
    }
    // Return column values.
    snprintf(info, InfoLen,
             "%*c %*d %*d %*d %*d %*d",
             InfoSpacing, param[PARAM_UPLO].c,
             InfoSpacing, param[PARAM_N].i,
             InfoSpacing, param[PARAM_NRHS].i,
             InfoSpacing, param[PARAM_PADA].i,
             InfoSpacing, param[PARAM_PADB].i,
             InfoSpacing, param[PARAM_NB].i);

    //================================================================
    // Set parameters.
    //================================================================
    PLASMA_enum uplo = PLASMA_uplo_const(param[PARAM_UPLO].c);

    int n = param[PARAM_N].i;
    int nrhs = param[PARAM_NRHS].i;

    int Am, An, Bm, Bn;

    Am = n;
    An = n;
    Bm = n;
    Bn = nrhs;

    int lda = imax(1, Am + param[PARAM_PADA].i);
    int ldb = imax(1, Bm + param[PARAM_PADB].i);

    int test = param[PARAM_TEST].c == 'y';
    double tol = param[PARAM_TOL].d * LAPACKE_dlamch('E');

    //================================================================
    // Set tuning parameters.
    //================================================================
    PLASMA_Set(PLASMA_TILE_SIZE, param[PARAM_NB].i);

    //================================================================
    // Allocate and initialize arrays.
    //================================================================
    PLASMA_Complex64_t *A =
        (PLASMA_Complex64_t*)malloc((size_t)lda*An*sizeof(PLASMA_Complex64_t));
    assert(A != NULL);

    PLASMA_Complex64_t *B =
        (PLASMA_Complex64_t*)malloc((size_t)ldb*Bn*sizeof(PLASMA_Complex64_t));
    assert(B != NULL);

    int seed[] = {0, 0, 0, 1};
    lapack_int retval;
    retval = LAPACKE_zlarnv(1, seed, (size_t)lda*An, A);
    assert(retval == 0);

    retval = LAPACKE_zlarnv(1, seed, (size_t)lda*Bn, B);
    assert(retval == 0);

    //================================================================
    // Make the A matrix symmetric/Hermitian positive definite.
    // It increases diagonal by n, and makes it real.
    // It sets Aji = conj( Aij ) for j < i, that is, copy lower
    // triangle to upper triangle.
    //================================================================
    for (int i = 0; i < n; ++i) {
        A(i,i) = creal(A(i,i)) + n;
        for (int j = 0; j < i; ++j) {
            A(j,i) = conj(A(i,j));
        }
    }

    PLASMA_Complex64_t *Aref = NULL;
    PLASMA_Complex64_t *Bref = NULL;
    double *work = NULL;
    if (test) {
        Aref = (PLASMA_Complex64_t*)malloc(
            (size_t)lda*An*sizeof(PLASMA_Complex64_t));
        assert(Aref != NULL);

        Bref = (PLASMA_Complex64_t*)malloc(
            (size_t)ldb*Bn*sizeof(PLASMA_Complex64_t));
        assert(Bref != NULL);

        work = (double*)malloc((size_t)n*sizeof(double));
        assert(work != NULL);

        memcpy(Aref, A, (size_t)lda*An*sizeof(PLASMA_Complex64_t));
        memcpy(Bref, B, (size_t)ldb*Bn*sizeof(PLASMA_Complex64_t));
    }

    //================================================================
    // Run POTRF
    //================================================================
    PLASMA_zpotrf(uplo, n, A, lda);

    //================================================================
    // Run and time PLASMA.
    //================================================================
    plasma_time_t start = omp_get_wtime();
    PLASMA_zpotrs(uplo, n, nrhs, A, lda, B, ldb);
    plasma_time_t stop = omp_get_wtime();
    plasma_time_t time = stop-start;
    double flops = flops_zpotrf(n) + 2*flops_ztrsm(uplo, n, nrhs);
    param[PARAM_TIME].d = time;
    param[PARAM_GFLOPS].d = flops / time / 1e9;

    //================================================================
    // Test results by checking the residual (backward error)
    //================================================================
    if (test) {
        PLASMA_Complex64_t zone  =   1.0;
        PLASMA_Complex64_t zmone =  -1.0;

        double Anorm = LAPACKE_zlange_work(
                           LAPACK_COL_MAJOR, 'I', Am, An, Aref, lda, work);
        double Xnorm = LAPACKE_zlange_work(
                           LAPACK_COL_MAJOR, 'I', Bm, Bn, B,    ldb, work);

        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, nrhs, n,
                        CBLAS_SADDR(zmone), Aref, lda,
                                            B, ldb,
                        CBLAS_SADDR(zone), Bref, ldb);

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
    if (test) {
        free(Aref);
        free(Bref);
        free(work);
    }
}
