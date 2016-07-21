/**
 *
 * @file test_zpotrf.c
 *
 *  PLASMA test routine.
 *  PLASMA is a software package provided by Univ. of Tennessee,
 *  Univ. of California Berkeley, Univ. of Colorado Denver and
 *  Univ. of Manchester.
 *
 * @version 3.0.0
 * @author Pedro V. Lara
 * @date
 * @precisions normal z -> s d c
 *
 **/
#include "core_blas.h"
#include "test.h"
#include "flops.h"

#include <assert.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef PLASMA_WITH_MKL
    #include <mkl_cblas.h>
    #include <mkl_lapacke.h>
#else
    #include <cblas.h>
    #include <lapacke.h>
#endif
#include <omp.h>
#include <plasma.h>

#define COMPLEX

#define A(i,j)  A[i + j*lda]

/***************************************************************************//**
 *
 * @brief Tests ZPOTRF.
 *
 * @param[in]  param - array of parameters
 * @param[out] info  - string of column labels or column values; length InfoLen
 *
 * If param is NULL     and info is NULL,     print usage and return.
 * If param is NULL     and info is non-NULL, set info to column headings and return.
 * If param is non-NULL and info is non-NULL, set info to column values   and run test.
 ******************************************************************************/
void test_zpotrf(param_value_t param[], char *info)
{
    //================================================================
    // Print usage info or return column labels or values.
    //================================================================
    if (param == NULL) {
        if (info == NULL) {
            // Print usage info.
            print_usage(PARAM_UPLO);
            print_usage(PARAM_N);
            print_usage(PARAM_PADA);
        }
        else {
            // Return column labels.
            snprintf(info, InfoLen,
                "%*s %*s %*s",
                InfoSpacing, "Uplo",
                InfoSpacing, "N",
                InfoSpacing, "PadA");
        }
        return;
    }
    // Return column values.
    snprintf(info, InfoLen,
        "%*c %*d %*d",
        InfoSpacing, param[PARAM_UPLO].c,
        InfoSpacing, param[PARAM_N].i,
        InfoSpacing, param[PARAM_PADA].i);

    //================================================================
    // Set parameters.
    //================================================================
    PLASMA_enum uplo;

    if (param[PARAM_UPLO].c == 'l')
        uplo = PlasmaLower;
    else
        uplo = PlasmaUpper;

    int n = param[PARAM_N].i;

    int Am, An;

    Am = n;
    An = n;

    int lda = imax(1, Am + param[PARAM_PADA].i);

    int test = param[PARAM_TEST].c == 'y';
    double tol = param[PARAM_TOL].d * LAPACKE_dlamch('E');

    //================================================================
    // Allocate and initialize arrays.
    //================================================================
    PLASMA_Complex64_t *A =
        (PLASMA_Complex64_t*)malloc((size_t)lda*An*sizeof(PLASMA_Complex64_t));
    assert(A != NULL);

    int seed[] = {0, 0, 0, 1};
    lapack_int retval;
    retval = LAPACKE_zlarnv(1, seed, (size_t)lda*An, A);
    assert(retval == 0);

    //================================================================
    // Make the A matrix symmetric/Hermitian positive definite.
    // It increases diagonal by n, and makes it real.
    // It sets Aji = conj( Aij ) for j < i, that is, copy lower
    // triangle to upper triangle.
    //================================================================
    int i, j;
    for (i=0; i < n; ++i) {
        A(i,i) = (creal(A(i,i)) + n) + 0. * I;
        for (j=0; j < i; ++j) {
            A(j,i) = conj(A(i,j));
        }
    }

    PLASMA_Complex64_t *Aref = NULL;
    if (test) {
        Aref = (PLASMA_Complex64_t*)malloc(
            (size_t)lda*An*sizeof(PLASMA_Complex64_t));
        assert(Aref != NULL);

        memcpy(Aref, A, (size_t)lda*An*sizeof(PLASMA_Complex64_t));
    }

    //================================================================
    // Run and time PLASMA.
    //================================================================
    plasma_time_t start = omp_get_wtime();
    PLASMA_zpotrf((CBLAS_UPLO)uplo, n, A, lda);
    plasma_time_t stop = omp_get_wtime();
    plasma_time_t time = stop-start;

    param[PARAM_TIME].d = time;
    param[PARAM_GFLOPS].d = flops_zpotrf(n) / time / 1e9;

    //================================================================
    // Test results by comparing to a reference implementation.
    //================================================================
    if (test) {
        LAPACKE_zpotrf(
            LAPACK_COL_MAJOR,
            lapack_const(uplo), n,
            Aref, lda);

        PLASMA_Complex64_t zmone = -1.0;
        cblas_zaxpy((size_t)lda*An, CBLAS_SADDR(zmone), Aref, 1, A, 1);

        double work[1];
        double Anorm = LAPACKE_zlange_work(
                           LAPACK_COL_MAJOR, 'F', Am, An, Aref, lda, work);
        double error = LAPACKE_zlange_work(
                           LAPACK_COL_MAJOR, 'F', Am, An, A,    lda, work);
        if (Anorm != 0)
            error /= Anorm;

        param[PARAM_ERROR].d = error;
        param[PARAM_SUCCESS].i = error < tol;
    }

    //================================================================
    // Free arrays.
    //================================================================
    free(A);
    if (test)
        free(Aref);
}
