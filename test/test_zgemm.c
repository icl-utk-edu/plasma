/**
 *
 * @file test_zgemm.c
 *
 *  PLASMA test routines.
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

void test_zgemm(param_value_t param[])
{
    if (param == NULL) {
        print_usage(PARAM_TRANSA);
        print_usage(PARAM_TRANSB);
        print_usage(PARAM_M);
        print_usage(PARAM_N);
        print_usage(PARAM_K);
        print_usage(PARAM_PADA);
        print_usage(PARAM_PADB);
        print_usage(PARAM_PADC);
        return;
    }

    PLASMA_enum transa;
    PLASMA_enum transb;

    if (param[PARAM_TRANSA].i == 'n')
        transa = PlasmaNoTrans;
    else if (param[PARAM_TRANSA].i == 't')
        transa = PlasmaTrans;
    else
        transa = PlasmaConjTrans;

    if (param[PARAM_TRANSB].i == 'n')
        transb = PlasmaNoTrans;
    else if (param[PARAM_TRANSB].i == 't')
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

    int lda = Am + param[PARAM_PADA].i;
    int ldb = Bm + param[PARAM_PADB].i;
    int ldc = Cm + param[PARAM_PADC].i;

    int test = param[PARAM_TEST].i;

    PLASMA_Complex64_t *A =
        (PLASMA_Complex64_t*)malloc((size_t)lda*An*sizeof(PLASMA_Complex64_t));
    assert(A != NULL);

    PLASMA_Complex64_t *B =
        (PLASMA_Complex64_t*)malloc((size_t)ldb*An*sizeof(PLASMA_Complex64_t));
    assert(B != NULL);

    PLASMA_Complex64_t *C1 =
        (PLASMA_Complex64_t*)malloc((size_t)ldc*Cn*sizeof(PLASMA_Complex64_t));
    assert(C1 != NULL);

    PLASMA_Complex64_t *C2;
    if (test) {
        C2 = (PLASMA_Complex64_t*)malloc(
            (size_t)ldc*Cn*sizeof(PLASMA_Complex64_t));
        assert(C2 != NULL);
    }

    PLASMA_Complex64_t alpha = -1.0;
    PLASMA_Complex64_t beta = 1.0;

    double start = omp_get_wtime();
    cblas_zgemm(
        CblasColMajor,
        (CBLAS_TRANSPOSE)transa, (CBLAS_TRANSPOSE)transb,
        m, n, k,
        CBLAS_SADDR(alpha), A, lda,
                B, ldb,
         CBLAS_SADDR(beta), C1, ldc);
    double stop = omp_get_wtime();

    if (test) {
        memcpy(C2, C1, (size_t)ldc*Cn*sizeof(PLASMA_Complex64_t));

        cblas_zgemm(
            CblasColMajor,
            (CBLAS_TRANSPOSE)transa, (CBLAS_TRANSPOSE)transb,
            m, n, k,
            CBLAS_SADDR(alpha), A, lda,
                    B, ldb,
             CBLAS_SADDR(beta), C2, ldc);





    }

}
