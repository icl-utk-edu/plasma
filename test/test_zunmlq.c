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
#include "plasma.h"
#include "core_lapack.h"

#include <assert.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <omp.h>

#define COMPLEX

/***************************************************************************//**
 *
 * @brief Tests ZUNMLQ.
 *
 * @param[in,out] param - array of parameters
 * @param[in]     run - whether to run test
 *
 * Sets flags in param indicating which parameters are used.
 * If run is true, also runs test and stores output parameters.
 ******************************************************************************/
void test_zunmlq(param_value_t param[], bool run)
{
    //================================================================
    // Mark which parameters are used.
    //================================================================
    param[PARAM_SIDE   ].used = true;
    param[PARAM_TRANS  ].used = true;
    param[PARAM_DIM    ].used = PARAM_USE_M | PARAM_USE_N | PARAM_USE_K;
    param[PARAM_PADA   ].used = true;
    param[PARAM_PADB   ].used = true;
    param[PARAM_NB     ].used = true;
    param[PARAM_IB     ].used = true;
    param[PARAM_HMODE  ].used = true;
    if (! run)
        return;

    //================================================================
    // Set parameters.
    //================================================================
    plasma_enum_t trans = plasma_trans_const(param[PARAM_TRANS].c);
    plasma_enum_t side  = plasma_side_const(param[PARAM_SIDE].c);

    int m = param[PARAM_DIM].dim.m;
    int n = param[PARAM_DIM].dim.n;

    // Number of Householder reflectors to use.
    int k = param[PARAM_DIM].dim.k;

    // Dimensions of matrix A differ for different combinations of
    // side and trans.
    int am, an;
    if (side == PlasmaLeft) {
        an = m;
        if (trans == PlasmaNoTrans) {
            am = k;
        }
        else {
            am = m;
        }
    }
    else {
        an = n;
        if (trans == PlasmaNoTrans) {
            am = n;
        }
        else {
            am = k;
        }
    }
    int lda = imax(1, am + param[PARAM_PADA].i);

    // Dimensions of matrix B.
    int bm = m;
    int bn = n;
    int ldb = imax(1, bm  + param[PARAM_PADB].i);

    int test = param[PARAM_TEST].c == 'y';
    double tol = param[PARAM_TOL].d * LAPACKE_dlamch('E');

    //================================================================
    // Set tuning parameters.
    //================================================================
    plasma_set(PlasmaTuning, PlasmaDisabled);
    plasma_set(PlasmaNb, param[PARAM_NB].i);
    plasma_set(PlasmaIb, param[PARAM_IB].i);
    if (param[PARAM_HMODE].c == 't')
        plasma_set(PlasmaHouseholderMode, PlasmaTreeHouseholder);
    else
        plasma_set(PlasmaHouseholderMode, PlasmaFlatHouseholder);

    //================================================================
    // Allocate and initialize array A for construction of matrix Q as
    // A = L*Q.
    //================================================================
    plasma_complex64_t *A =
        (plasma_complex64_t*)malloc((size_t)lda*an*sizeof(plasma_complex64_t));
    assert(A != NULL);

    int seed[] = {0, 0, 0, 1};
    lapack_int retval;
    retval = LAPACKE_zlarnv(1, seed, (size_t)lda*an, A);
    assert(retval == 0);

    //================================================================
    // Prepare factorization of matrix A.
    //================================================================
    plasma_desc_t T;
    plasma_zgelqf(am, an, A, lda, &T);

    //================================================================
    // Prepare m-by-n matrix B.
    //================================================================
    plasma_complex64_t *B =
        (plasma_complex64_t*)malloc((size_t)ldb*bn*
                                            sizeof(plasma_complex64_t));
    assert(B != NULL);

    retval = LAPACKE_zlarnv(1, seed, (size_t)ldb*bn, B);
    assert(retval == 0);

    plasma_complex64_t *Bref = NULL;
    if (test) {
        // Store the original array if residual is to be evaluated.
        Bref = (plasma_complex64_t*)malloc((size_t)ldb*bn*
                                           sizeof(plasma_complex64_t));
        assert(Bref != NULL);

        memcpy(Bref, B, (size_t)ldb*bn*sizeof(plasma_complex64_t));
    }

    //================================================================
    // Prepare explicit matrix Q.
    //================================================================
    // Number of Householder reflectors to be used depends on
    // side and trans combination.
    int qk = am;

    int qm, qn, ldq;
    plasma_complex64_t *Q = NULL;
    if (test) {
        qm  = am;
        qn  = an;
        ldq = qm;
        Q = (plasma_complex64_t *)malloc((size_t)ldq*qn*
                                         sizeof(plasma_complex64_t));
        // Build explicit Q.
        plasma_zunglq(qm, qn, qk, A, lda, T, Q, ldq);
    }

    //================================================================
    // Run and time PLASMA.
    //================================================================
    plasma_time_t start = omp_get_wtime();
    plasma_zunmlq(side, trans,
                  bm, bn, qk,
                  A, lda, T,
                  B, ldb);
    plasma_time_t stop = omp_get_wtime();
    plasma_time_t time = stop-start;

    param[PARAM_TIME].d = time;
    param[PARAM_GFLOPS].d = flops_zunmlq(side, bm, bn, qk) /
                            time / 1e9;

    //================================================================
    // Test results by comparing implicit and explicit actions of Q.
    //================================================================
    if (test) {
        // Set dimensions of the resulting matrix C = op(Q)*B or C = B*op(Q).
        int cm, cn;
        if (side == PlasmaLeft) {
            cn = bn;
            if (trans == PlasmaNoTrans) {
                cm = qm;
            }
            else {
                cm = qn;
            }
        }
        else {
            cm = bm;
            if (trans == PlasmaNoTrans) {
                cn = qn;
            }
            else {
                cn = qm;
            }
        }

        // |Q*B|_1
        double work[1];
        double normC = LAPACKE_zlange_work(LAPACK_COL_MAJOR, '1', cm, cn,
                                           B, ldb, work);


        // Apply explicit Q and compute the difference. For example, for
        // PlasmaLeft and PlasmaNoTrans, B <- implicit(Q)*B - Q*Bref.
        if (side == PlasmaLeft) {
            plasma_zgemm(trans, PlasmaNoTrans,
                         cm, cn, m,
                         -1.0, Q, ldq,
                               Bref, ldb,
                          1.0, B, ldb);
        }
        else {
            plasma_zgemm(PlasmaNoTrans, trans,
                         cm, cn, n,
                         -1.0, Bref, ldb,
                               Q, ldq,
                          1.0, B, ldb);
        }

        // Compute error in the difference.
        // |implicit(Q)*B - Q*Bref|_1
        double error = LAPACKE_zlange_work(LAPACK_COL_MAJOR, '1', cm, cn,
                                           B, ldb, work);

        // Normalize the result.
        error /= (cm * normC);

        // Store the results.
        param[PARAM_ERROR].d = error;
        param[PARAM_SUCCESS].i = (error < tol);
    }


    //================================================================
    // Free arrays.
    //================================================================
    plasma_desc_destroy(&T);
    free(A);
    free(B);
    if (test) {
        free(Bref);
        free(Q);
    }
}
