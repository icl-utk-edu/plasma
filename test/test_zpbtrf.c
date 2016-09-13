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
 * @brief Tests ZPBTRF.
 *
 * @param[in]  param - array of parameters
 * @param[out] info  - string of column labels or column values; length InfoLen
 *
 * If param is NULL and info is NULL,     print usage and return.
 * If param is NULL and info is non-NULL, set info to column labels and return.
 * If param is non-NULL and info is non-NULL, set info to column values
 * and run test.
 ******************************************************************************/
void test_zpbtrf(param_value_t param[], char *info)
{
    //================================================================
    // Print usage info or return column labels or values.
    //================================================================
    if (param == NULL) {
        if (info == NULL) {
            // Print usage info.
            //  pbtrf params
            print_usage(PARAM_UPLO);
            print_usage(PARAM_N);
            print_usage(PARAM_KL);
            print_usage(PARAM_PADA);
            print_usage(PARAM_NB);
            //  gbtrs params for check
            print_usage(PARAM_NRHS);
            print_usage(PARAM_PADB);
        }
        else {
            // Return column labels.
            snprintf(info, InfoLen,
                "%*s %*s %*s %*s %*s %*s %*s ",
                InfoSpacing, "UpLo",
                InfoSpacing, "N",
                InfoSpacing, "KD",
                InfoSpacing, "PadA",
                InfoSpacing, "NB",
                InfoSpacing, "NRHS",
                InfoSpacing, "PadB");
        }
        return;
    }
    // Return column values.
    snprintf(info, InfoLen,
        "%*c %*d %*d %*d %*d %*d %*d",
        InfoSpacing, param[PARAM_UPLO].c,
        InfoSpacing, param[PARAM_N].i,
        InfoSpacing, param[PARAM_KL].i,
        InfoSpacing, param[PARAM_PADA].i,
        InfoSpacing, param[PARAM_NB].i,
        InfoSpacing, param[PARAM_NRHS].i,
        InfoSpacing, param[PARAM_PADB].i);

    //================================================================
    // Set parameters.
    //================================================================
    plasma_enum_t uplo = PLASMA_uplo_const(param[PARAM_UPLO].c);
    int pada = param[PARAM_PADA].i;
    int n    = param[PARAM_N].i;
    int lda  = imax(1, n + pada);

    int kd   = (uplo == PlasmaUpper ? param[PARAM_KU].i : param[PARAM_KL].i);

    int test = param[PARAM_TEST].c == 'y';
    double tol = param[PARAM_TOL].d * LAPACKE_dlamch('E');

    //================================================================
    // Set tuning parameters.
    //================================================================
    PLASMA_Set(PLASMA_TILE_SIZE, param[PARAM_NB].i);

    //================================================================
    // Allocate and initialize arrays.
    //================================================================
    // band matrix A in full storage (also used for solution check)
    plasma_complex64_t *A =
        (plasma_complex64_t*)malloc((size_t)lda*n*sizeof(plasma_complex64_t));
    assert(A != NULL);

    int seed[] = {0, 0, 0, 1};
    lapack_int retval;
    retval = LAPACKE_zlarnv(1, seed, (size_t)lda*n, A);
    assert(retval == 0);
    // make it SPD
    int i, j;
    for (i = 0; i < n; ++i) {
        A(i,i) = creal(A(i,i)) + n;
        for (j = 0; j < i; ++j) {
            A(j,i) = conj(A(i,j));
        }
    }
    // zero out elements outside the band
    plasma_complex64_t zzero = 0.0;
    for (i = 0; i < n; i++) {
        for (j = i+kd+1; j < n; j++) A(i, j) = zzero;
    }
    for (j = 0; j < n; j++) {
        for (i = j+kd+1; i < n; i++) A(i, j) = zzero;
    }

    // band matrix A in LAPACK storage
    int ldab = imax(1, kd+1+pada);
    plasma_complex64_t *AB = NULL;
    AB = (plasma_complex64_t*)malloc((size_t)ldab*n*sizeof(plasma_complex64_t));
    assert(AB != NULL);

    // convert into LAPACK's symmetric band storage
    for (j = 0; j < n; j++) {
        for (i = 0; i < ldab; i++) AB[i + j*ldab] = zzero;
        if (uplo == PlasmaUpper) {
            for (i = imax(0, j-kd); i <= j; i++) AB[i-j+kd + j*ldab] = A(i, j);
        }
        else {
            for (i = j; i <= imin(n-1, j+kd); i++) AB[i-j + j*ldab] = A(i, j);
        }
    }

    //================================================================
    // Run and time PLASMA.
    //================================================================
    int iinfo;

    plasma_time_t start = omp_get_wtime();
    iinfo = PLASMA_zpbtrf(uplo, n, kd, AB, ldab);
    if (iinfo != 0) printf( " zpbtrf failed with info=%d\n", iinfo );
    plasma_time_t stop = omp_get_wtime();
    plasma_time_t time = stop-start;

    param[PARAM_TIME].d = time;
    param[PARAM_GFLOPS].d = 0.0 / time / 1e9;

    //================================================================
    // Test results by computing residual norm.
    //================================================================
    if (test) {
        plasma_complex64_t zone =   1.0;
        plasma_complex64_t zmone = -1.0;

        int nrhs = param[PARAM_NRHS].i;
        int ldb = imax(1, n + param[PARAM_PADB].i);

        // set up right-hand-side B
        plasma_complex64_t *B = NULL;
        B = (plasma_complex64_t*)malloc((size_t)ldb*nrhs*sizeof(plasma_complex64_t));
        assert(B != NULL);

        retval = LAPACKE_zlarnv(1, seed, (size_t)ldb*nrhs, B);
        assert(retval == 0);

        // copy B to X
        int ldx = ldb;
        plasma_complex64_t *X = NULL;
        X = (plasma_complex64_t*)malloc((size_t)ldx*nrhs*sizeof(plasma_complex64_t));
        assert(X != NULL);
        LAPACKE_zlacpy_work(LAPACK_COL_MAJOR, 'F', n, nrhs, B, ldb, X, ldx);

        // solve for X
        iinfo = PLASMA_zpbtrs(uplo, n, kd, nrhs, AB, ldab, X, ldb);
        if (iinfo != 0) printf( " zpbtrs failed with info = %d\n", iinfo );

        // compute residual vector
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, nrhs, n,
                    CBLAS_SADDR(zmone), A, lda,
                                        X, ldx,
                    CBLAS_SADDR(zone),  B, ldb);

        // compute various norms
        double *work = NULL;
        work = (double*)malloc((size_t)n*sizeof(double));
        assert(work != NULL);

        double Anorm = LAPACKE_zlange_work(LAPACK_COL_MAJOR, 'F', n, n,    A, lda, work);
        double Xnorm = LAPACKE_zlange_work(LAPACK_COL_MAJOR, 'I', n, nrhs, X, ldb, work);
        double Rnorm = LAPACKE_zlange_work(LAPACK_COL_MAJOR, 'I', n, nrhs, B, ldb, work);
        double residual = Rnorm/(n*Anorm*Xnorm);

        param[PARAM_ERROR].d = residual;
        param[PARAM_SUCCESS].i = residual < tol;

        // free arrays
        free(work);
        free(X);
        free(B);
    }

    //================================================================
    // Free arrays.
    //================================================================
    free(AB);
    free(A);
}
