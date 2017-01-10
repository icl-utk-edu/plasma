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
#include <math.h>

#include <omp.h>

#define COMPLEX

#define A(i_, j_) A[(i_) + (size_t)lda*(j_)]

/***************************************************************************//**
 *
 * @brief Tests ZTRMM
 *
 * @param[in]  param - array of parameters
 * @param[out] info  - string of column labels or column values; length InfoLen
 *
 * If param is NULL     and info is NULL,     print usage and return.
 * If param is NULL     and info is non-NULL, set info to column headings
 * and return.
 * If param is non-NULL and info is non-NULL, set info to column values
 * and run test.
 ******************************************************************************/
void test_ztrmm(param_value_t param[], char *info)
{
    //================================================================
    // Print usage info or return column labels or values
    //================================================================
    if (param == NULL) {
        if (info == NULL) {
            // Print usage info
            print_usage(PARAM_SIDE);
            print_usage(PARAM_UPLO);
            print_usage(PARAM_TRANSA);
            print_usage(PARAM_DIAG);
            print_usage(PARAM_M);
            print_usage(PARAM_N);
            print_usage(PARAM_ALPHA);
            print_usage(PARAM_PADA);
            print_usage(PARAM_PADB);
        }
        else {
            // Return column labels
            snprintf(info, InfoLen,
                "%*s %*s %*s %*s %*s %*s %*s %*s %*s",
                InfoSpacing, "Side",
                InfoSpacing, "UpLo",
                InfoSpacing, "TransA",
                InfoSpacing, "Diag",
                InfoSpacing, "m",
                InfoSpacing, "n",
                InfoSpacing, "alpha",
                InfoSpacing, "PadA",
                InfoSpacing, "PadB");
        }
        return;
    }
    // Return column values
    snprintf(info, InfoLen,
        "%*c %*c %*c %*c %*d %*d %*.4f %*d %*d",
        InfoSpacing, param[PARAM_SIDE].c,
        InfoSpacing, param[PARAM_UPLO].c,
        InfoSpacing, param[PARAM_TRANSA].c,
        InfoSpacing, param[PARAM_DIAG].c,
        InfoSpacing, param[PARAM_M].i,
        InfoSpacing, param[PARAM_N].i,
        InfoSpacing, creal(param[PARAM_ALPHA].z),
        InfoSpacing, param[PARAM_PADA].i,
        InfoSpacing, param[PARAM_PADB].i);

    //================================================================
    // Set parameters
    //================================================================
    plasma_enum_t side = plasma_side_const(param[PARAM_SIDE].c);
    plasma_enum_t uplo = plasma_uplo_const(param[PARAM_UPLO].c);
    plasma_enum_t transa = plasma_trans_const(param[PARAM_TRANSA].c);
    plasma_enum_t diag = plasma_diag_const(param[PARAM_DIAG].c);

    int m = param[PARAM_M].i;
    int n = param[PARAM_N].i;

    int k;
    int lda;

    if (side == PlasmaLeft) {
        k    = m;
        lda  = imax(1, m + param[PARAM_PADA].i);
    }
    else {
        k    = n;
        lda  = imax(1, n + param[PARAM_PADA].i);
    }

    int    ldb  = imax(1, m + param[PARAM_PADB].i);
    int    test = param[PARAM_TEST].c == 'y';
    double eps  = LAPACKE_dlamch('E');

#ifdef COMPLEX
    plasma_complex64_t alpha = param[PARAM_ALPHA].z;
#else
    double alpha = creal(param[PARAM_ALPHA].z);
#endif

    //================================================================
    // Set tuning parameters.
    //================================================================
    plasma_set(PlasmaNb, param[PARAM_NB].i);

    //================================================================
    // Allocate and initialize arrays.
    //================================================================
    plasma_complex64_t *A =
        (plasma_complex64_t *)malloc((size_t)lda*k*sizeof(plasma_complex64_t));
    assert(A != NULL);

    plasma_complex64_t *B =
        (plasma_complex64_t *)malloc((size_t)ldb*n*sizeof(plasma_complex64_t));
    assert(B != NULL);

    int seed[] = {0, 0, 0, 1};
    lapack_int retval;
    retval = LAPACKE_zlarnv(1, seed, (size_t)lda*k, A);
    assert(retval == 0);

    retval = LAPACKE_zlarnv(1, seed, (size_t)ldb*n, B);
    assert(retval == 0);

    plasma_complex64_t *Bref = NULL;
    if (test) {
        Bref = (plasma_complex64_t*)malloc(
            (size_t)ldb*n*sizeof(plasma_complex64_t));
        assert(Bref != NULL);

        memcpy(Bref, B, (size_t)ldb*n*sizeof(plasma_complex64_t));
    }

    //================================================================
    // Run and time PLASMA.
    //================================================================
    plasma_time_t start = omp_get_wtime();

    plasma_ztrmm(side, uplo,
                 transa, diag,
                 m, n, alpha, A, lda, B, ldb);

    plasma_time_t stop = omp_get_wtime();
    plasma_time_t time = stop-start;

    param[PARAM_TIME].d   = time;
    param[PARAM_GFLOPS].d = flops_ztrmm(side, m, n) / time / 1e9;

    //================================================================
    // Test results by comparing to a reference implementation.
    //================================================================
    if (test) {
        // see comments in test_zgemm.c
        plasma_complex64_t zmone = -1.0;
        double work[1];

        // LAPACKE_[ds]lantr_work has a bug (returns 0)
        // in MKL <= 11.3.3 (at least). Fixed in LAPACK 3.6.1.
        // For now, call LAPACK directly.
        // LAPACK_zlantr is a macro for correct name mangling (e.g.
        // adding _ at the end) of the Fortran symbol.
        // The macro is either defined in lapacke.h, or in the file
        // core_lapack_z.h for the use with MKL.
        char normc = 'F';
        char uploc = lapack_const(uplo);
        char diagc = lapack_const(diag);
        double Anorm = LAPACK_zlantr(&normc, &uploc, &diagc,
                                     &k, &k, A, &lda, work);
        //double Anorm = LAPACKE_zlantr_work(
        //                   LAPACK_COL_MAJOR, 'F', lapack_const(uplo),
        //                   lapack_const(diag), k, k, A, lda, work);

        double Bnorm = LAPACKE_zlange_work(
                           LAPACK_COL_MAJOR, 'F', m, n, Bref, ldb, work);

        cblas_ztrmm(CblasColMajor, (CBLAS_SIDE)side, (CBLAS_UPLO)uplo,
                   (CBLAS_TRANSPOSE)transa, (CBLAS_DIAG)diag,
                    m, n, CBLAS_SADDR(alpha), A, lda, Bref, ldb);

        cblas_zaxpy((size_t)ldb*n, CBLAS_SADDR(zmone), Bref, 1, B, 1);

        double error = LAPACKE_zlange_work(
                           LAPACK_COL_MAJOR, 'F', m, n, B,    ldb, work);
        double normalize = sqrt((double)k+2) * cabs(alpha) * Anorm * Bnorm;
        if (normalize != 0)
            error /= normalize;

        param[PARAM_ERROR].d = error;
        param[PARAM_SUCCESS].i = error < 3*eps;
    }

    //================================================================
    // Free arrays.
    //================================================================
    free(A);
    free(B);
    if (test)
        free(Bref);
}
