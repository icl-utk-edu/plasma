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

#define A(i_, j_) A[(i_) + (size_t)lda*(j_)]

/***************************************************************************//**
 *
 * @brief Tests ZHETRF_AA.
 *
 * @param[in]  param - array of parameters
 * @param[out] info  - string of column labels or column values; length InfoLen
 *
 * If param is NULL and info is NULL,     print usage and return.
 * If param is NULL and info is non-NULL, set info to column labels and return.
 * If param is non-NULL and info is non-NULL, set info to column values
 * and run test.
 ******************************************************************************/
void test_zhetrf(param_value_t param[], char *info)
{
    //================================================================
    // Print usage info or return column labels or values.
    //================================================================
    if (param == NULL) {
        if (info == NULL) {
            // Print usage info.
            print_usage(PARAM_UPLO);
            print_usage(PARAM_DIM);
            print_usage(PARAM_PADA);
            print_usage(PARAM_NRHS);
            print_usage(PARAM_NB);
            print_usage(PARAM_NTPF);
        }
        else {
            // Return column labels.
            snprintf(info, InfoLen,
                     "%*s %*s %*s %*s %*s %*s",
                     InfoSpacing, "Uplo",
                     InfoSpacing, "N",
                     InfoSpacing, "PadA",
                     InfoSpacing, "NRHS",
                     InfoSpacing, "NB",
                     InfoSpacing, "NTPF");
        }
        return;
    }
    // Return column values.
    snprintf(info, InfoLen,
             "%*c %*d %*d %*d %*d %*d",
             InfoSpacing, param[PARAM_UPLO].c,
             InfoSpacing, param[PARAM_DIM].dim.n,
             InfoSpacing, param[PARAM_PADA].i,
             InfoSpacing, param[PARAM_NRHS].i,
             InfoSpacing, param[PARAM_NB].i,
             InfoSpacing, param[PARAM_NTPF].i);

    //================================================================
    // Set parameters.
    //================================================================
    plasma_enum_t uplo = plasma_uplo_const(param[PARAM_UPLO].c);

    int n = param[PARAM_DIM].dim.n;
    int nb = param[PARAM_NB].i;

    int lda = imax(1, n + param[PARAM_PADA].i);
    // band matrix A in skewed LAPACK storage
    int kut  = (nb+nb+nb-1)/nb; // # of tiles in upper band (not including diagonal)
    int klt  = (nb+nb-1)/nb;    // # of tiles in lower band (not including diagonal)
    int ldt  = (kut+klt+1)*nb;  // since we use zgetrf on panel, we pivot back within panel.

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
        (plasma_complex64_t*)calloc((size_t)lda*n, sizeof(plasma_complex64_t));
    assert(A != NULL);
    plasma_complex64_t *T =
        (plasma_complex64_t*)calloc((size_t)ldt*n, sizeof(plasma_complex64_t));
    assert(T != NULL);

    int *ipiv = (int*)calloc((size_t)n, sizeof(int));
    int *ipiv2 = (int*)calloc((size_t)n, sizeof(int));
    assert(ipiv != NULL);

    int seed[] = {0, 0, 0, 1};
    lapack_int retval;
    retval = LAPACKE_zlarnv(1, seed, (size_t)lda*n, A);
    assert(retval == 0);

    //================================================================
    // Make the A matrix Hermitian.
    // It sets Aji = conj( Aij ) for j < i, that is, copy lower
    // triangle to upper triangle.
    //================================================================
    for (int i = 0; i < n; ++i) {
        A(i,i) = creal(A(i,i));
        for (int j = 0; j < i; ++j) {
            A(j,i) = conj(A(i,j));
        }
    }

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
    int plainfo = plasma_zhetrf(uplo, n, A, lda, ipiv, T, ldt, ipiv2);
    plasma_time_t stop = omp_get_wtime();
    plasma_time_t time = stop-start;

    param[PARAM_TIME].d = time;
    param[PARAM_GFLOPS].d = flops_zpotrf(n) / time / 1e9;

    //================================================================
    // Test results by comparing to a reference implementation.
    //================================================================
    if (test && plainfo == 0) {
        // compute the residual norm ||A-bx||
        int nrhs = param[PARAM_NRHS].i;
        int ldb = imax(1, n + param[PARAM_PADB].i);
        int ldx = ldb;

        plasma_complex64_t *B = (plasma_complex64_t*)malloc(
            (size_t)ldb*nrhs*sizeof(plasma_complex64_t));
        assert(B != NULL);
        plasma_complex64_t *X = (plasma_complex64_t*)malloc(
            (size_t)ldx*nrhs*sizeof(plasma_complex64_t));
        assert(X != NULL);

        // set up right-hand-side B = A*rand
        retval = LAPACKE_zlarnv(1, seed, (size_t)ldb*nrhs, X);
        assert(retval == 0);
        plasma_zgemm(PlasmaNoTrans, PlasmaNoTrans,
                     n, nrhs, n,
                     1.0, Aref, lda,
                             X, ldx,
                     0.0,    B, ldb);

        // copy B to X
        LAPACKE_zlacpy_work(
            LAPACK_COL_MAJOR, 'F', n, nrhs, B, ldb, X, ldx);

        // solve for X
        int iinfo = plasma_zhetrs(
            PlasmaNoTrans, n, nrhs, A, lda, ipiv, T, ldt, ipiv2, X, ldx);
        if (iinfo != 0) printf( " zhetrs failed, info = %d\n", iinfo );

        // compute residual vector
        plasma_zgemm(PlasmaNoTrans, PlasmaNoTrans,
                     n, nrhs, n,
                     -1.0, Aref, lda,
                              X, ldx,
                      1.0,    B, ldb);

        // compute various norms
        double *work = NULL;
        work = (double*)malloc((size_t)n*sizeof(double));
        assert(work != NULL);

        double Anorm = LAPACKE_zlange_work(
            LAPACK_COL_MAJOR, 'F', n, n,    Aref, lda, work);
        double Xnorm = LAPACKE_zlange_work(
            LAPACK_COL_MAJOR, 'I', n, nrhs, X, ldb, work);
        double Rnorm = LAPACKE_zlange_work(
            LAPACK_COL_MAJOR, 'I', n, nrhs, B, ldb, work);
        double residual = Rnorm/(n*Anorm*Xnorm);

        param[PARAM_ERROR].d = residual;
        param[PARAM_SUCCESS].i = residual < tol;

        // free workspaces
        free(work);
        free(X);
        free(B);
    }

    //================================================================
    // Free arrays.
    //================================================================
    free(A); free(ipiv); free(ipiv2);
    free(T);
    if (test)
        free(Aref);
}
