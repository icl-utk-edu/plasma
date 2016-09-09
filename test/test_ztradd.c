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
static int save_vec(int n, PLASMA_Complex64_t a[], FILE *f);
static int save_mtrx(int m, int n, PLASMA_Complex64_t A[], FILE *f);
static int ext_tria_mtrx(int m, int n, char uplo, PLASMA_Complex64_t A[]);
static int vec_cpy(int n, PLASMA_Complex64_t a[], PLASMA_Complex64_t b[]);
static int mtrx_cpy(int m, int n, PLASMA_Complex64_t A[], PLASMA_Complex64_t B[]);
static int mtrx_tran(bool conjFlag, int m, int n, PLASMA_Complex64_t A[]);

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

    //================================================================
    // @test Save random matrix A
    //================================================================
    FILE *f;
    f = fopen("A0.mtrx", "w");
    save_mtrx(Am, An, A, f);
    fclose(f);

    //================================================================
    // @test Save random matrix B
    //================================================================
    f = fopen("B0.mtrx", "w");
    save_mtrx(Bm, Bn, B, f);
    fclose(f);

    //================================================================

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
    param[PARAM_GFLOPS].d = flops_ztradd(m, n) / time / 1e9;

    //================================================================
    // @test Save result matrix B
    //================================================================
    f = fopen("B.mtrx", "w");
    save_mtrx(Bm, Bn, B, f);
    fclose(f);

    //================================================================
    // Test results by comparing to a matrix addition in a for loop
    //================================================================
    if (test) {
        // Calculate relative error |B_ref - B|_F / |B_ref|_F < 3*eps
        // Using 3*eps covers complex arithmetic

        //================================================================
        // @test Save random matrix A
        //================================================================
        f = fopen("A.mtrx", "w");
        save_mtrx(An, Am, A, f);
        fclose(f);

        // Make matrix A upper or lower triangular
        if (uplo == PlasmaUpper) {
            ext_tria_mtrx(Am, An, 'u', A);

            //================================================================
            // @test Save upper triangular matrix A
            //================================================================
            f = fopen("Au.mtrx", "w");
            save_mtrx(An, Am, A, f);
            fclose(f);
        }
        else if (uplo == PlasmaLower) {
            ext_tria_mtrx(Am, An, 'l', A);

            //================================================================
            // @test Save lower triangular matrix A
            //================================================================
            f = fopen("Al.mtrx", "w");
            save_mtrx(An, Am, A, f);
            fclose(f);
        }

        // Transpose matrix A
        if (transA == PlasmaTrans) {
            mtrx_tran(false, Am, An, A);

            //================================================================
            // @test Save transposed matrix A
            //================================================================
            f = fopen("At.mtrx", "w");
            save_mtrx(An, Am, A, f);
            fclose(f);
        }
        else if (transA == PlasmaConjTrans) {
            mtrx_tran(true, Am, An, A);

            //================================================================
            // @test Save conjugate transposed matrix A
            //================================================================
            f = fopen("Ac.mtrx", "w");
            save_mtrx(An, Am, A, f);
            fclose(f);
        }

        for (int i = 0; i < Bm; i++) {
            for (int j = 0; j < Bn; j++) {

                Bref[i*Bn+j] = alpha*A[i*Bn+j] + beta*Bref[i*Bn+j];

            }
        }

        //================================================================
        // @test Save result matrix Bref
        //================================================================
        f = fopen("Bref.mtrx", "w");
        save_mtrx(Bm, Bn, Bref, f);
        fclose(f);

        double work[1];

        // Calculate Frobenius norm of reference result B_ref
        double BnormRef  = LAPACKE_zlange_work(
                               LAPACK_COL_MAJOR, 'F', Bm, Bn, Bref, ldb, work);

        // Calculate difference B_ref-B
        PLASMA_Complex64_t zmone = -1.0;
        cblas_zaxpy((size_t)ldb*Bn, CBLAS_SADDR(zmone), B, 1, Bref, 1);

        //================================================================
        // @test Save difference matrix B_ref-B
        //================================================================
        f = fopen("D.mtrx", "w");
        save_mtrx(Bm, Bn, Bref, f);
        fclose(f);

        // Calculate Frobenius norm of B_ref-B
        double BnormDiff = LAPACKE_zlange_work(
                               LAPACK_COL_MAJOR, 'F', Bm, Bn, Bref, ldb, work);

        // Calculate relative error |B_ref-B|_F / |B_ref|_F
        double error = BnormDiff/BnormRef;
    
        printf("|B_ref-B|_F: %g\t|B_ref|_F: %g\terror: %g\n",
            BnormDiff, BnormRef, error);

        param[PARAM_ERROR].d   = error;
        param[PARAM_SUCCESS].i = error < 3*eps;
    }

    //================================================================
    // Free arrays
    //================================================================
    free(A);
    free(B);

    if (test)
        free(Bref);
}

// Saves contents of vector to given file
static int save_vec(int n, PLASMA_Complex64_t a[], FILE *f) {

  int i;

  for (i = 0; i < n; i++) {

    // fprintf(f, "%g\t", a[i]);
    fprintf(f, "%.3f%+.3fi\t", creal(a[i]), cimag(a[i]));

  }

  fprintf(f, "\n");

  return 0;

}

// Saves contents of matrix to given file
static int save_mtrx(int m, int n, PLASMA_Complex64_t A[], FILE *f) {

  int i;

  for (i = 0; i < m*n; i = i+n) {

    save_vec(n, &A[i], f);

  }

  return 0;

}

// Extracts triagonal matrix: upper or lower from given matrix A
static int ext_tria_mtrx(int m, int n, char uplo, PLASMA_Complex64_t A[]) {

    // Erase lower triangualr part, preserve upper triangular part
    // incl. main diagonal
    if (uplo == 'u' || uplo == 'U') {

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < i; j++) {
                A[i*n+j] = 0.0;
            }
        }
    }
    // Erase upper triangualr part, preserve lower triangular part
    // incl. main diagonal
    else if (uplo == 'l' || uplo == 'L') {

        for (int i = 0; i < m-1; i++) {
            for (int j = i+1; j < n; j++) {
                A[i*n+j] = 0.0;
            }
        }
    }
    return 0;
}

// Copies one vector into another vector
static int vec_cpy(int n, PLASMA_Complex64_t a[], PLASMA_Complex64_t b[]) {

  int i;

  for (i = 0; i < n; i++) {

    b[i] = a[i];

  }

  return 0;

}

// Copies one matrix into another
static int mtrx_cpy(int m, int n, PLASMA_Complex64_t A[], PLASMA_Complex64_t B[]) {

  int i;

  for (i = 0; i < m; i++) {

    vec_cpy(n, &A[i*n], &B[i*n]);

  }

  return 0;

}

// Transposes matrix
static int mtrx_tran(bool conjFlag, int m, int n, PLASMA_Complex64_t A[]) {

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
