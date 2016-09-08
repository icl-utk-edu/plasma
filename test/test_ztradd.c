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

#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

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

// Declarations of local functions
int vec_cpy(int n, PLASMA_Complex64_t a[], PLASMA_Complex64_t b[]);
int mtrx_cpy(int m, int n, PLASMA_Complex64_t A[], PLASMA_Complex64_t B[]);
int mtrx_tran(bool conjFlag, int m, int n, PLASMA_Complex64_t A[]);

/***************************************************************************//**
 *
 * @brief Tests ZTRADD
 *
 * @param[in]  param - array of parameters
 * @param[out] info  - string of column labels or column values; length InfoLen
 *
 * If param is NULL and info is NULL,     print usage and return.
 * If param is NULL and info is non-NULL, set info to column labels and return.
 * If param is non-NULL and info is non-NULL, set info to column values
 * and run test.
 ******************************************************************************/
void test_ztradd(param_value_t param[], char *info)
{
    //================================================================
    // Print usage info or return column labels or values
    //================================================================
    if (param == NULL) {
        if (info == NULL) {
            // Print usage info
            print_usage(PARAM_UPLO);
            print_usage(PARAM_TRANSA);
            print_usage(PARAM_M);
            print_usage(PARAM_N);
            print_usage(PARAM_ALPHA);
            print_usage(PARAM_BETA);
            print_usage(PARAM_PADA);
            print_usage(PARAM_PADB);
            print_usage(PARAM_NB);
        }
        else {
            // Return column labels
            snprintf(info, InfoLen,
                     "%*s %*s %*s %*s %*s %*s %*s %*s %*s",
                     InfoSpacing, "UpLo",
                     InfoSpacing, "TransA",
                     InfoSpacing, "m",
                     InfoSpacing, "n",
                     InfoSpacing, "alpha",
                     InfoSpacing, "beta",
                     InfoSpacing, "PadA",
                     InfoSpacing, "PadB",
                     InfoSpacing, "nb");
        }
        return;
    }
    // Return column values
    snprintf(info, InfoLen,
             "%*c %*c %*d %*d %*.4f %*.4f %*d %*d %*d",
             InfoSpacing, param[PARAM_UPLO].c,
             InfoSpacing, param[PARAM_TRANSA].c,
             InfoSpacing, param[PARAM_M].i,
             InfoSpacing, param[PARAM_N].i,
             InfoSpacing, creal(param[PARAM_ALPHA].z),
             InfoSpacing, creal(param[PARAM_BETA].z),
             InfoSpacing, param[PARAM_PADA].i,
             InfoSpacing, param[PARAM_PADB].i,
             InfoSpacing, param[PARAM_NB].i);

    //================================================================
    // Set parameters
    //================================================================
    PLASMA_enum uplo;
    PLASMA_enum transA;

    if (param[PARAM_UPLO].c == 'f')
        uplo = PlasmaFull;
    else if (param[PARAM_UPLO].c == 'u')
        uplo = PlasmaUpper;
    else
        uplo = PlasmaLower;

    if (param[PARAM_TRANSA].c == 'n')
        transA = PlasmaNoTrans;
    else if (param[PARAM_TRANSA].c == 't')
        transA = PlasmaTrans;
    else
        transA = PlasmaConjTrans;

    int m = param[PARAM_M].i;
    int n = param[PARAM_N].i;

    int Am, An;
    int Bm, Bn;

    if (transA == PlasmaNoTrans) {
        Am = m;
        An = n;
    }
    else {
        Am = n;
        An = m;
    }

    Bm = m;
    Bn = n;

    int lda = imax(1, Am + param[PARAM_PADA].i);
    int ldb = imax(1, Bm + param[PARAM_PADB].i);

    int    test = param[PARAM_TEST].c == 'y';
    double eps  = LAPACKE_dlamch('E');

    //================================================================
    // Set tuning parameters
    //================================================================
    PLASMA_Set(PLASMA_TILE_SIZE, param[PARAM_NB].i);

    //================================================================
    // Allocate and initialize arrays
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

    retval = LAPACKE_zlarnv(1, seed, (size_t)ldb*Bn, B);
    assert(retval == 0);

    PLASMA_Complex64_t *Bref = NULL;
    if (test) {
        Bref = (PLASMA_Complex64_t*)malloc(
            (size_t)ldb*Bn*sizeof(PLASMA_Complex64_t));
        assert(Bref != NULL);

        memcpy(Bref, B, (size_t)ldb*Bn*sizeof(PLASMA_Complex64_t));
    }

#ifdef COMPLEX
    PLASMA_Complex64_t alpha = param[PARAM_ALPHA].z;
    PLASMA_Complex64_t beta  = param[PARAM_BETA].z;
#else
    double alpha = creal(param[PARAM_ALPHA].z);
    double beta  = creal(param[PARAM_BETA].z);
#endif

    //================================================================
    // Run and time PLASMA
    //================================================================
    plasma_time_t start = omp_get_wtime();

    PLASMA_ztradd((CBLAS_UPLO)uplo, (CBLAS_TRANSPOSE)transA, m, n,
                   alpha, A, lda, beta, B, ldb);

    plasma_time_t stop = omp_get_wtime();
    plasma_time_t time = stop-start;

    param[PARAM_TIME].d   = time;
    // @todo Create correct formula for TRADD FLOP/s calculation
    // param[PARAM_GFLOPS].d = flops_ztradd(m, n) / time / 1e9;
    param[PARAM_GFLOPS].d = 0.0;

    //================================================================
    // Test results by comparing to a matrix addition in a for loop
    //================================================================
    if (test) {
        // |R - R_ref|_p < gamma_{k+2} * |alpha| * |A|_p * |B|_p +
        //                 gamma_2 * |beta| * |C|_p
        // holds component-wise or with |.|_p as 1, inf, or Frobenius norm.
        // gamma_k = k*eps / (1 - k*eps), but we use
        // gamma_k = sqrt(k)*eps as a statistical average case.
        // Using 3*eps covers complex arithmetic.
        // See Higham, Accuracy and Stability of Numerical Algorithms, ch 2-3.

        // Transpose matrix A
        if (transA == PlasmaTrans) {
            mtrx_tran(false, Am, An, A);
        }
        else if (transA == PlasmaConjTrans) {
            mtrx_tran(true, Am, An, A);
        }

        for (int i = 0; i < Bm; i++) {
            for (int j = 0; j < Bn; j++) {

                Bref[i*Bn+j] = alpha*A[i*Bn+j] + beta*Bref[i*Bn+j];

            }
        }

        PLASMA_Complex64_t zmone = -1.0;
        cblas_zaxpy((size_t)ldb*Bn, CBLAS_SADDR(zmone), Bref, 1, B, 1);

        double work[1];
        double Anorm = LAPACKE_zlange_work(
                           LAPACK_COL_MAJOR, 'F', Am, An, A,    lda, work);
        double Bnorm = LAPACKE_zlange_work(
                           LAPACK_COL_MAJOR, 'F', Bm, Bn, B,    ldb, work);
        double Cnorm = LAPACKE_zlange_work(
                           LAPACK_COL_MAJOR, 'F', Cm, Cn, Cref, ldc, work);
        double error = LAPACKE_zlange_work(
                           LAPACK_COL_MAJOR, 'F', Cm, Cn, C,    ldc, work);
        double normalize = sqrt((double)k+2) * cabs(alpha) * Anorm * Bnorm
                         + 2 * cabs(beta) * Cnorm;
        if (normalize != 0)
            error /= normalize;

        param[PARAM_ERROR].d   = error;
        param[PARAM_SUCCESS].i = error < 3*eps;
    }

    //================================================================
    // Free arrays
    //================================================================
    free(A);
    free(B);
    free(C);
    if (test)
        free(Cref);
}

// Copies one vector into another vector
int vec_cpy(int n, PLASMA_Complex64_t a[], PLASMA_Complex64_t b[]) {

  int i;

  for (i = 0; i < n; i++) {

    b[i] = a[i];

  }

  return 0;

}

// Copies one matrix into another
int mtrx_cpy(int m, int n, PLASMA_Complex64_t A[], PLASMA_Complex64_t B[]) {

  int i;

  for (i = 0; i < m; i++) {

    vec_cpy(n, &A[i*n], &B[i*n]);

  }

  return 0;

}

// Transposes matrix
int mtrx_tran(bool conjFlag, int m, int n, PLASMA_Complex64_t A[]) {

  int i, j;

  PLASMA_Complex64_t *T = calloc(m*n, sizeof(PLASMA_Complex64_t));

  if (T==NULL) {

    fprintf(stderr, "'calloc()' failed!\n");
    return(-1);

  }

  for (i = 0; i < m; i++) {
    for (j = 0; j < n; j++) {
      
      if (conjFlag == true)
        T[j*m+i] = conj(A[i*n+j]);
      else
        T[j*m+i] = A[i*n+j];

    }
  }

  mtrx_cpy(m, n, T, A);

  free(T);

  return 0;

}
