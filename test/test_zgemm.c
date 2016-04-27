/**
 *
 * @file test_zgemm.c
 *
 *  PLASMA test routine.
 *  PLASMA is a software package provided by Univ. of Tennessee,
 *  Univ. of California Berkeley and Univ. of Colorado Denver.
 *
 * @version 3.0.0
 * @author Jakub Kurzak
 * @date 2016-01-01
 * @precisions normal z -> s d c
 *
 **/
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

/***************************************************************************//**
 *
 * @brief Tests ZGEMM.
 *
 * @param[in]  param - array of parameters
 * @param[out] info  - string of column labels or column values; length InfoLen
 *
 * If param is NULL     and info is NULL,     print usage and return.
 * If param is NULL     and info is non-NULL, set info to column headings and return.
 * If param is non-NULL and info is non-NULL, set info to column values   and run test.
 ******************************************************************************/
void test_zgemm(param_value_t param[], char *info)
{
    //================================================================
    // Print usage info or return column labels or values.
    //================================================================
    if (param == NULL) {
        if (info == NULL) {
            // Print usage info.
            print_usage(PARAM_TRANSA);
            print_usage(PARAM_TRANSB);
            print_usage(PARAM_M);
            print_usage(PARAM_N);
            print_usage(PARAM_K);
            print_usage(PARAM_PADA);
            print_usage(PARAM_PADB);
            print_usage(PARAM_PADC);
        }
        else {
            // Return column labels.
            snprintf(info, InfoLen,
                   "%*s%*s%*s%*s%*s%*s%*s%*s",
                   InfoSpacing, "TransA",
                   InfoSpacing, "TransB",
                   InfoSpacing, "M",
                   InfoSpacing, "N",
                   InfoSpacing, "K",
                   InfoSpacing, "PadA",
                   InfoSpacing, "PadB",
                   InfoSpacing, "PadC");
        }
        return;
    }
    // Return column values.
    snprintf(info, InfoLen,
             "%*c%*c%*d%*d%*d%*d%*d%*d",
             InfoSpacing, param[PARAM_TRANSA].c,
             InfoSpacing, param[PARAM_TRANSB].c,
             InfoSpacing, param[PARAM_M].i,
             InfoSpacing, param[PARAM_N].i,
             InfoSpacing, param[PARAM_K].i,
             InfoSpacing, param[PARAM_PADA].i,
             InfoSpacing, param[PARAM_PADB].i,
             InfoSpacing, param[PARAM_PADC].i);

    //================================================================
    // Set parameters.
    //================================================================
    PLASMA_enum transa;
    PLASMA_enum transb;

    if (param[PARAM_TRANSA].c == 'n')
        transa = PlasmaNoTrans;
    else if (param[PARAM_TRANSA].c == 't')
        transa = PlasmaTrans;
    else
        transa = PlasmaConjTrans;

    if (param[PARAM_TRANSB].c == 'n')
        transb = PlasmaNoTrans;
    else if (param[PARAM_TRANSB].c == 't')
        transb = PlasmaTrans;
    else
        transb = PlasmaConjTrans;

    int m = param[PARAM_M].i;
    int n = param[PARAM_N].i;
    int k = param[PARAM_K].i;

    int Am, An;
    int Bm, Bn;
    int Cm, Cn;

    if (transa == PlasmaNoTrans) {
        Am = m;
        An = k;
    }
    else {
        Am = k;
        An = m;
    }
    if (transb == PlasmaNoTrans) {
        Bm = k;
        Bn = n;
    }
    else {
        Bm = n;
        Bn = k;
    }
    Cm = m;
    Cn = n;

    int lda = imax(1, Am + param[PARAM_PADA].i);
    int ldb = imax(1, Bm + param[PARAM_PADB].i);
    int ldc = imax(1, Cm + param[PARAM_PADC].i);

    int test = param[PARAM_TEST].c == 'y';
    double tol = param[PARAM_TOL].d * LAPACKE_dlamch('E');

    //================================================================
    // Allocate and initialize arrays.
    //================================================================
    PLASMA_Complex64_t *A =
        (PLASMA_Complex64_t*)malloc((size_t)lda*An*sizeof(PLASMA_Complex64_t));
    assert(A != NULL);

    PLASMA_Complex64_t *B =
        (PLASMA_Complex64_t*)malloc((size_t)ldb*Bn*sizeof(PLASMA_Complex64_t));
    assert(B != NULL);

    PLASMA_Complex64_t *C1 =
        (PLASMA_Complex64_t*)malloc((size_t)ldc*Cn*sizeof(PLASMA_Complex64_t));
    assert(C1 != NULL);

    int seed[] = {0, 0, 0, 1};
    lapack_int retval;
    retval = LAPACKE_zlarnv(1, seed, (size_t)lda*An, A);
    assert(retval == 0);

    retval = LAPACKE_zlarnv(1, seed, (size_t)ldb*Bn, B);
    assert(retval == 0);

    retval = LAPACKE_zlarnv(1, seed, (size_t)ldc*Cn, C1);
    assert(retval == 0);

    PLASMA_Complex64_t *C2 = NULL;
    if (test) {
        C2 = (PLASMA_Complex64_t*)malloc(
            (size_t)ldc*Cn*sizeof(PLASMA_Complex64_t));
        assert(C2 != NULL);

        memcpy(C2, C1, (size_t)ldc*Cn*sizeof(PLASMA_Complex64_t));
    }

    PLASMA_Complex64_t alpha = (PLASMA_Complex64_t)1.234;
    PLASMA_Complex64_t beta = (PLASMA_Complex64_t)-5.678;

    //================================================================
    // Run and time PLASMA.
    //================================================================
    plasma_time_t start = omp_get_wtime();
    PLASMA_zgemm(
        (CBLAS_TRANSPOSE)transa, (CBLAS_TRANSPOSE)transb,
        m, n, k,
        alpha, A, lda,
               B, ldb,
         beta, C1, ldc);
    plasma_time_t stop = omp_get_wtime();
    plasma_time_t time = stop-start;

    param[PARAM_TIME].d = time;
    param[PARAM_GFLOPS].d = flops_zgemm(m, n, k) / time / 1e9;

    //================================================================
    // Test results by comparing to a reference implementation.
    //================================================================
    if (test) {
        cblas_zgemm(
            CblasColMajor,
            (CBLAS_TRANSPOSE)transa, (CBLAS_TRANSPOSE)transb,
            m, n, k,
            CBLAS_SADDR(alpha), A, lda,
                                B, ldb,
             CBLAS_SADDR(beta), C2, ldc);

        PLASMA_Complex64_t zmone = (PLASMA_Complex64_t)-1.0;
        cblas_zaxpy((size_t)ldc*Cn, CBLAS_SADDR(zmone), C1, 1, C2, 1);

        double Cnorm = LAPACKE_zlange(LAPACK_COL_MAJOR, 'F', Cm, Cn, C1, ldc);
        double error = LAPACKE_zlange(LAPACK_COL_MAJOR, 'F', Cm, Cn, C2, ldc);
        if (Cnorm != 0)
            error /= Cnorm;

        param[PARAM_ERROR].d = error;
        param[PARAM_SUCCESS].i = error < tol;
    }

    //================================================================
    // Free arrays.
    //================================================================
    free(A);
    free(B);
    free(C1);
    if (test)
        free(C2);
}
