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

/******************************************************************************/
void plasma_zgetrf__(int m, int n, plasma_complex64_t *A, int lda, int *ipiv, int nb)
{
    for (int k = 0; k < imin(m, n); k += nb) {

        int kb = imin(imin(m, n)-k, nb);

        // panel
        LAPACKE_zgetrf(LAPACK_COL_MAJOR,
                       m-k,
                       kb,
                       &A[k+k*lda], lda,
                       &ipiv[k]);

        for (int i = k+1; i <= imin(m, k+kb); i++)
            ipiv[i-1] += k;

        // right pivoting
        LAPACKE_zlaswp(LAPACK_COL_MAJOR,
                       n-k-kb,
                       &A[(k+kb)*lda], lda,
                       k+1,
                       k+kb,
                       ipiv, 1);

        plasma_complex64_t zone = 1.0;
        cblas_ztrsm(CblasColMajor,
                    PlasmaLeft, PlasmaLower,
                    PlasmaNoTrans, PlasmaUnit,
                    kb,
                    n-k-kb,
                    CBLAS_SADDR(zone), &A[k+k*lda], lda,
                                       &A[k+(k+kb)*lda], lda);

        plasma_complex64_t zmone = -1.0;
        cblas_zgemm(CblasColMajor,
                    CblasNoTrans, CblasNoTrans,
                    m-k-kb,
                    n-k-kb,
                    kb,
                    CBLAS_SADDR(zmone), &A[k+kb+k*lda], lda,
                                        &A[k+(k+kb)*lda], lda,
                    CBLAS_SADDR(zone),  &A[(k+kb)+(k+kb)*lda], lda);
    }

    // left pivoting
    for (int k = nb; k < imin(m, n); k += nb) {

        LAPACKE_zlaswp(LAPACK_COL_MAJOR,
                       nb,
                       &A[(k-nb)*lda], lda,
                       k+1,
                       imin(m, n),
                       ipiv, 1);
    }
}

/******************************************************************************/
void plasma_zgetrf_(int m, int n, plasma_complex64_t *A, int lda, int *ipiv, int nb)
{
    for (int k = 0; k < imin(m, n); k++) {

        // panel
        LAPACKE_zgetrf(LAPACK_COL_MAJOR,
                       m-k, 1, &A[k+k*lda], lda, &ipiv[k]);

        for (int i = k+1; i <= imin(m, k+1); i++)
            ipiv[i-1] += k;

        // left pivoting
        LAPACKE_zlaswp(LAPACK_COL_MAJOR,
                       k,
                       &A[0], lda,
                       k+1,
                       k+1,
                       ipiv, 1);

        if (k == imin(m, n)-1)
            return;

        int l = (k+1)&(~k);

        // right pivoting
        LAPACKE_zlaswp(LAPACK_COL_MAJOR,
                       l,
                       &A[(k+1)*lda], lda,
                       (k-l+1)+1,
                       k+1,
                       ipiv, 1);

        plasma_complex64_t zone = 1.0;
        cblas_ztrsm(CblasColMajor,
                    PlasmaLeft, PlasmaLower,
                    PlasmaNoTrans, PlasmaUnit,
                    l,
                    l,
                    CBLAS_SADDR(zone), &A[k+1-l+(k+1-l)*lda], lda,
                                       &A[k+1-l+(k+1)*lda], lda);

        plasma_complex64_t zmone = -1.0;
        cblas_zgemm(CblasColMajor,
                    CblasNoTrans, CblasNoTrans,
                    m-k-1,
                    l,
                    l,
                    CBLAS_SADDR(zmone), &A[k+1+(k+1-l)*lda], lda,
                                        &A[k+1-l+(k+1)*lda], lda,
                    CBLAS_SADDR(zone),  &A[k+1+(k+1)*lda], lda);
    }

}

/******************************************************************************/
static void print_matrix(plasma_complex64_t *A, int m, int n)
{
    for (int j = 0; j < m; j++) {
        for (int i = 0; i < n; i++) {

            double v = cabs(A[j+i*m]);
            char c;

                 if (v < 0.0000000001) c = '.';
            else if (v == 1.0) c = '#';
            else c = 'o';

            printf ("%c ", c);
        }
        printf("\n");
    }
}

#define COMPLEX

#define A(i_, j_) A[(i_) + (size_t)lda*(j_)]

/***************************************************************************//**
 *
 * @brief Tests ZPOTRF.
 *
 * @param[in]  param - array of parameters
 * @param[out] info  - string of column labels or column values; length InfoLen
 *
 * If param is NULL and info is NULL,     print usage and return.
 * If param is NULL and info is non-NULL, set info to column labels and return.
 * If param is non-NULL and info is non-NULL, set info to column values
 * and run test.
 ******************************************************************************/
void test_zgetrf(param_value_t param[], char *info)
{
    //================================================================
    // Print usage info or return column labels or values.
    //================================================================
    if (param == NULL) {
        if (info == NULL) {
            // Print usage info.
            print_usage(PARAM_M);
            print_usage(PARAM_N);
            print_usage(PARAM_PADA);
            print_usage(PARAM_NB);
        }
        else {
            // Return column labels.
            snprintf(info, InfoLen,
                     "%*s %*s %*s %*s",
                     InfoSpacing, "M",
                     InfoSpacing, "N",
                     InfoSpacing, "PadA",
                     InfoSpacing, "NB");
        }
        return;
    }
    // Return column values.
    snprintf(info, InfoLen,
             "%*d %*d %*d %*d",
             InfoSpacing, param[PARAM_N].i,
             InfoSpacing, param[PARAM_N].i,
             InfoSpacing, param[PARAM_PADA].i,
             InfoSpacing, param[PARAM_NB].i);

    //================================================================
    // Set parameters.
    //================================================================
    int m = param[PARAM_M].i;
    int n = param[PARAM_N].i;

    int lda = imax(1, m+param[PARAM_PADA].i);

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

    int *IPIV = (int*)malloc((size_t)m*sizeof(int));
    assert(IPIV != NULL);

    int seed[] = {0, 0, 0, 1};
    lapack_int retval;
    retval = LAPACKE_zlarnv(1, seed, (size_t)lda*n, A);
    assert(retval == 0);

// for (int i = 0; i < imin(m, n); i++)
//     A[i*lda+i] = 10000.0;

    plasma_complex64_t *Aref = NULL;
    if (test) {
        Aref = (plasma_complex64_t*)malloc(
            (size_t)lda*n*sizeof(plasma_complex64_t));
        assert(Aref != NULL);

        memcpy(Aref, A, (size_t)lda*n*sizeof(plasma_complex64_t));
    }

    //================================================================
    // Run and time PLASMA.
    //================================================================
    plasma_time_t start = omp_get_wtime();
    plasma_zgetrf_(m, n, A, lda, IPIV, param[PARAM_NB].i);
    plasma_time_t stop = omp_get_wtime();
    plasma_time_t time = stop-start;

    param[PARAM_TIME].d = time;
    param[PARAM_GFLOPS].d = flops_zpotrf(n) / time / 1e9;

    //================================================================
    // Test results by comparing to a reference implementation.
    //================================================================
    if (test) {
        LAPACKE_zgetrf(
            LAPACK_COL_MAJOR,
            m, n,
            Aref, lda, IPIV);

        plasma_complex64_t zmone = -1.0;
        cblas_zaxpy((size_t)lda*n, CBLAS_SADDR(zmone), Aref, 1, A, 1);

print_matrix(A, m, n);

        double work[1];
        double Anorm = LAPACKE_zlange_work(
            LAPACK_COL_MAJOR, 'F', m, n, Aref, lda, work);

        double error = LAPACKE_zlange_work(
            LAPACK_COL_MAJOR, 'F', m, n, A, lda, work);

        if (Anorm != 0)
            error /= Anorm;

        param[PARAM_ERROR].d = error;
        param[PARAM_SUCCESS].i = error < tol;
    }

    //================================================================
    // Free arrays.
    //================================================================
    free(A);
    free(IPIV);
    if (test)
        free(Aref);
}
