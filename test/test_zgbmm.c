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
 /* Implemented by Daniel Mishler beginning on 07-12-12, 11:05 */
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
 * @brief Tests ZGEMM.
 *
 * @param[in,out] param - array of parameters
 * @param[in]     run - whether to run test
 *
 * Sets used flags in param indicating parameters that are used.
 * If run is true, also runs test and stores output parameters.
 ******************************************************************************/
void test_zgbmm(param_value_t param[], bool run)
{
    //================================================================
    // Mark which parameters are used.
    //================================================================
    param[PARAM_TRANSA ].used = true;
    param[PARAM_TRANSB ].used = true;
    param[PARAM_DIM    ].used = PARAM_USE_M | PARAM_USE_N | PARAM_USE_K;
    param[PARAM_ALPHA  ].used = true;
    param[PARAM_BETA   ].used = true;
    param[PARAM_PADA   ].used = true;
    param[PARAM_PADB   ].used = true;
    param[PARAM_PADC   ].used = true;
    param[PARAM_NB     ].used = true;
    param[PARAM_KL     ].used = true;
    param[PARAM_KU     ].used = true;
    if (! run)
        return;

    //================================================================
    // Set parameters.
    //================================================================
    plasma_enum_t transa = plasma_trans_const(param[PARAM_TRANSA].c);
    plasma_enum_t transb = plasma_trans_const(param[PARAM_TRANSB].c);

    int m = param[PARAM_DIM].dim.m;
    int n = param[PARAM_DIM].dim.n;
    int k = param[PARAM_DIM].dim.k;

    int kl = param[PARAM_KL].i;
    int ku = param[PARAM_KU].i;

    int Am, An;
    int Bm, Bn;
    int Cm, Cn;

    if (transa == PlasmaNoTrans) {
        Am = m;
        An = k;
    }
    else { // band matrix: swapping of KL and KU occurs later
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
    double eps = LAPACKE_dlamch('E');

#ifdef COMPLEX
    plasma_complex64_t alpha = param[PARAM_ALPHA].z;
    plasma_complex64_t beta  = param[PARAM_BETA].z;
#else
    double alpha = creal(param[PARAM_ALPHA].z);
    double beta  = creal(param[PARAM_BETA].z);
#endif

    //================================================================
    // Set tuning parameters.
    //================================================================
    plasma_set(PlasmaTuning, PlasmaDisabled);
    plasma_set(PlasmaNb, param[PARAM_NB].i);

    //================================================================
    // Allocate and initialize arrays.
    //================================================================
    plasma_complex64_t *A =
        (plasma_complex64_t*)malloc((size_t)lda*An*sizeof(plasma_complex64_t));
    assert(A != NULL);

    plasma_complex64_t *B =
        (plasma_complex64_t*)malloc((size_t)ldb*Bn*sizeof(plasma_complex64_t));
    assert(B != NULL);

    plasma_complex64_t *C =
        (plasma_complex64_t*)malloc((size_t)ldc*Cn*sizeof(plasma_complex64_t));
    assert(C != NULL);

    int seed[] = {0, 0, 0, 1};
    lapack_int retval;
    retval = LAPACKE_zlarnv(1, seed, (size_t)lda*An, A);
    assert(retval == 0);
    retval = LAPACKE_zlarnv(1, seed, (size_t)ldb*Bn, B);
    assert(retval == 0);

    retval = LAPACKE_zlarnv(1, seed, (size_t)ldc*Cn, C);
    assert(retval == 0);

    void naive_printmxn(plasma_complex64_t* M, int mRows, int nColumns)
    {
        int ii, jj;
        for(ii = 0; ii < mRows; ii++)
        {
            // finished a row
            printf("\n");
            for(jj = 0; jj < nColumns; jj++)
            {
                printf("%+1.3lf  ", M[jj*mRows+ii]);
            }
        }
        printf("\n");
    }
    // naive_printmxn(A,Am,An);
    // Set all values in A to zero to help debug
    // printf("Am=%d,An=%d,lda=%d\n",Am,An,lda);
    // printf("Bm=%d,Bn=%d,ldb=%d\n",Bm,Bn,ldb);
    // printf("Cm=%d,Cn=%d,ldc=%d\n",Cm,Cn,ldc);
    // This exact set of arguments sets a band matrix up *almost* perfectly
    // There is still the issue of corner bleeding.
    int cornerI, cornerJ, extralower, extraupper;
    if(kl+ku <= Am)
    {
        plasma_zlaset(PlasmaGeneral, Am-kl-ku, An-1, 0, 0, A+1+kl, lda+1);
        extralower = ku-1;
        extraupper = kl-1;
    }
    else
    {
        extralower = Am-1-kl;
        extraupper = Am-1-ku;
    }
    // naive_printmxn(A,Am,An);
    // fix the corner bleeding
    for(; extralower > 0; extralower--)
    {
        cornerI = Am-extralower;
        plasma_zlaset(PlasmaGeneral,1,extralower,0,0,A+cornerI,lda+1);
    }
    for(; extraupper > 0; extraupper--)
    {
        cornerJ = An-extraupper;
        plasma_zlaset(PlasmaGeneral,1,extraupper,0,0,A+lda*cornerJ,lda+1);
    }
    naive_printmxn(A,Am,An);
    naive_printmxn(B,Bm,Bn);
    naive_printmxn(C,Cm,Cn);
    printf("////////////////////////////\n");
    // Set all values in A to zero to help debug

    plasma_complex64_t *Cref = NULL;
    if (test) {
        Cref = (plasma_complex64_t*)malloc(
            (size_t)ldc*Cn*sizeof(plasma_complex64_t));
        assert(Cref != NULL);

        memcpy(Cref, C, (size_t)ldc*Cn*sizeof(plasma_complex64_t));
    }

    //================================================================
    // Run and time PLASMA.
    //================================================================
    plasma_time_t start = omp_get_wtime();

    /*
    plasma_zgemm(
        transa, transb,
        m, n, k,
        alpha, A, lda,
               B, ldb,
         beta, C, ldc);
    */
    plasma_zgbmm(
        transa, transb,
        m, n, k, kl, ku,
        alpha, A, lda,
               B, ldb,
         beta, C, ldc);
    plasma_time_t stop = omp_get_wtime();
    plasma_time_t time = stop-start;
    printf("////////////////////////////\n");
    naive_printmxn(C,Cm,Cn);

    param[PARAM_TIME].d = time;
    param[PARAM_GFLOPS].d = flops_zgemm(m, n, k) / time / 1e9;

    //================================================================
    // Test results by comparing to a reference implementation.
    //================================================================
    if (test) {
        // |R - R_ref|_p < gamma_{k+2} * |alpha| * |A|_p * |B|_p +
        //                 gamma_2 * |beta| * |C|_p
        // holds component-wise or with |.|_p as 1, inf, or Frobenius norm.
        // gamma_k = k*eps / (1 - k*eps), but we use
        // gamma_k = sqrt(k)*eps as a statistical average case.
        // Using 3*eps covers complex arithmetic.
        // See Higham, Accuracy and Stability of Numerical Algorithms, ch 2-3.
        double work[1];
        double Anorm = LAPACKE_zlange_work(
                           LAPACK_COL_MAJOR, 'F', Am, An, A,    lda, work);
        double Bnorm = LAPACKE_zlange_work(
                           LAPACK_COL_MAJOR, 'F', Bm, Bn, B,    ldb, work);
        double Cnorm = LAPACKE_zlange_work(
                           LAPACK_COL_MAJOR, 'F', Cm, Cn, Cref, ldc, work);

        // no need to compare to band matrix here unless we need to optimize
        // testing speeds
        cblas_zgemm( 
            CblasColMajor,
            (CBLAS_TRANSPOSE)transa, (CBLAS_TRANSPOSE)transb,
            m, n, k,
            CBLAS_SADDR(alpha), A, lda,
                                B, ldb,
             CBLAS_SADDR(beta), Cref, ldc);

        plasma_complex64_t zmone = -1.0;
        cblas_zaxpy((size_t)ldc*Cn, CBLAS_SADDR(zmone), Cref, 1, C, 1);

        double error = LAPACKE_zlange_work(
                           LAPACK_COL_MAJOR, 'F', Cm, Cn, C,    ldc, work);
        // ||C-Cref||f
        double normalize = sqrt((double)k+2) * cabs(alpha) * Anorm * Bnorm
                         + 2 * cabs(beta) * Cnorm;
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
    free(C);
    if (test)
        free(Cref);
}
