/**
 *
 * @file
 *
 *  PLASMA is a software package provided by:
 *  University of Tennessee, US,
 *  University of Manchester, UK.
 *
 * @precisions mixed zc -> ds
 *
 **/

#include "test.h"
#include "flops.h"
#include "core_blas.h"
#include "core_lapack.h"
#include "plasma.h"

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <omp.h>

// Use custom matrix library
// #include "ma-mtrx.h"

#define COMPLEX

/***************************************************************************//**
 *
 * @brief Tests ZCPOSV
 *
 * @param[in]  param - array of parameters
 * @param[out] info  - string of column labels or column values; length InfoLen
 *
 * If param is NULL and info is NULL, print usage and return.
 * If param is NULL and info is non-NULL, set info to column labels and return.
 * If param is non-NULL and info is non-NULL, set info to column values
 * and run test.
 ******************************************************************************/
void test_zcposv(param_value_t param[], char *info)
{
    //================================================================
    // Print usage info or return column labels or values
    //================================================================
    if (param == NULL) {
        if (info == NULL) {
            // Print usage info
            print_usage(PARAM_UPLO);
            print_usage(PARAM_N);
            print_usage(PARAM_NRHS);
            print_usage(PARAM_PADA);
            print_usage(PARAM_PADB);
        }
        else {
            // Return column labels
            snprintf(info, InfoLen,
                "%*s %*s %*s %*s %*s",
                InfoSpacing, "UpLo",
                InfoSpacing, "n",
                InfoSpacing, "nrhs",
                InfoSpacing, "PadA",
                InfoSpacing, "PadB");
        }
        return;
    }
    // Return column values
    snprintf(info, InfoLen,
        "%*c %*d %*d %*d %*d",
        InfoSpacing, param[PARAM_UPLO].c,
        InfoSpacing, param[PARAM_N].i,
        InfoSpacing, param[PARAM_NRHS].i,
        InfoSpacing, param[PARAM_PADA].i,
        InfoSpacing, param[PARAM_PADB].i);

    //================================================================
    // Set parameters
    //================================================================
    PLASMA_enum uplo;

    if (param[PARAM_UPLO].c == 'u')
        uplo = PlasmaUpper;
    else
        uplo = PlasmaLower;

    int n    = param[PARAM_N].i;
    int nrhs = param[PARAM_NRHS].i;
    int lda  = imax(1, n + param[PARAM_PADA].i);
    int ldb  = imax(1, n + param[PARAM_PADB].i);
    int ldx  = ldb;
    int ITER;

    int    test = param[PARAM_TEST].c == 'y';
    double eps  = LAPACKE_dlamch_work('e');

    //================================================================
    // Allocate and initialize arrays
    //================================================================
    PLASMA_Complex64_t *Aref =
        (PLASMA_Complex64_t *)malloc((size_t)lda*n*
                                      sizeof(PLASMA_Complex64_t));
    assert(Aref != NULL);

    PLASMA_Complex64_t *A =
        (PLASMA_Complex64_t *)malloc((size_t)lda*n*
                                      sizeof(PLASMA_Complex64_t));
    assert(A != NULL);

    PLASMA_Complex64_t *B =
        (PLASMA_Complex64_t *)malloc((size_t)ldb*nrhs*
                                      sizeof(PLASMA_Complex64_t));
    assert(B != NULL);

    PLASMA_Complex64_t *X =
        (PLASMA_Complex64_t *)malloc((size_t)ldx*nrhs*
                                      sizeof(PLASMA_Complex64_t));
    assert(X != NULL);

    /* Initialize A and A2 for Symmetric Positive Matrix (Hessenberg in the complex case) */
    /*
    PLASMA_zplghe( (double)N, N, A1, LDA, 51 );
    PLASMA_zlacpy( PlasmaUpperLower, N, N, A1, LDA, A2, LDA );
    */

    /* Initialize B1 and B2 */
    /*
    PLASMA_zplrnt( N, NRHS, B1, LDB, 371 );
    PLASMA_zlacpy( PlasmaUpperLower, N, NRHS, B1, LDB, B2, LDB );
    */

    // Initialise A(lda x n) as Symmetric Positive Definite matrix
    //(Hessenberg in case of complex precision)
    // Generate random matrix Aref
    int seed[] = {0, 0, 0, 1};
    lapack_int retval;
    retval = LAPACKE_zlarnv(1, seed, (size_t)lda*n, Aref);
    assert(retval == 0);

    // Make A symmetric: A = Aref*Aref'
    PLASMA_Complex64_t alpha = 1.0, beta = 0.0;

    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasTrans, lda, n, n,
                         CBLAS_SADDR(alpha), Aref, lda, Aref, lda,
                         CBLAS_SADDR(beta),  A,    lda);

    // Ensure diagonal dominance: A = A + n*eye(n)
    for (int i = 0; i < lda; i++) {
        A[i*n+i] *= lda;
    }

    // @test: Save matrix A into file
    // save_mtrx(lda, n, real(A), "A.mtrx");

    // Preserve a copy of the original SPD matrix A in Aref for test purposes
    if (test) {
        memcpy(Aref, A, (size_t)lda*n*sizeof(PLASMA_Complex64_t));
    }
    // Otherwise, deallocate memory
    else {
        free(Aref);
    }

    // Initialise random matrix B(ldb x nrhs)
    retval = LAPACKE_zlarnv(1, seed, (size_t)ldb*nrhs, B);
    assert(retval == 0);

    //================================================================
    // Run and time PLASMA
    //================================================================
    plasma_time_t start = omp_get_wtime();

// int PLASMA_zcposv(PLASMA_enum uplo, int n, int nrhs,
//                   PLASMA_Complex64_t *A, int lda,
//                   PLASMA_Complex64_t *B, int ldb,
//                   PLASMA_Complex64_t *X, int ldx, int *iter)

    PLASMA_zcposv(uplo, n, nrhs, A, lda, B, ldb, X, ldx, &ITER);

    plasma_time_t stop = omp_get_wtime();
    plasma_time_t time = stop-start;

    param[PARAM_TIME].d   = time;
    // @todo Develop correct formula for performance estimation of ZCPOSV
    // param[PARAM_GFLOPS].d = flops_zcposv(side, n, nrhs) / time / 1e9;
    param[PARAM_GFLOPS].d = 0.0;

    //================================================================
    // TESTING ZCPOSV
    //================================================================
    if (test) {
        double      Anorm, Bnorm, Xnorm, Rnorm, result;
        PLASMA_enum mtrxLayout = LAPACK_COL_MAJOR;
        char        mtrxNorm   = 'I';
    
        alpha =  1.0;
        beta  = -1.0;
    
        double *work = (double *)malloc(n*sizeof(double));

        /* Check the factorization and the solution */
        // info_solution = check_solution(N, NRHS, A1, LDA, B1, B2, LDB, eps);

        
        // Calculate infinite norms of matrices A, B, X
        Anorm = LAPACKE_zlange_work(mtrxLayout, mtrxNorm, n, n,    Aref,
                                    lda, work);

        Bnorm = LAPACKE_zlange_work(mtrxLayout, mtrxNorm, n, nrhs, B,
                                    ldb, work);

        Xnorm = LAPACKE_zlange_work(mtrxLayout, mtrxNorm, n, nrhs, X,
                                    ldx, work);
    
        // Calculate residual R = A*X-B, store result in B
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, nrhs, n,
                    CBLAS_SADDR(alpha), Aref, lda, X, ldx, CBLAS_SADDR(beta),
                    B, ldb);

        // Calculate infinite norm of residual matrix R
        Rnorm = LAPACKE_zlange_work(mtrxLayout, mtrxNorm, n, nrhs, B,
                                    ldb, work);
    
        // printf(" Solution obtained with %d iterations\n", ITER);
        // printf(Anorm, Bnorm, Xnorm, Rnorm);
    
        // Calculate normalized result
        result = Rnorm / ( (Anorm*Xnorm + Bnorm)*n*eps );
    
        param[PARAM_ERROR].d = result;

        if (isnan(Xnorm)  || isinf(Xnorm)  ||
            isnan(result) || isinf(result) || (result > 60.0)) {

            param[PARAM_SUCCESS].i = 0;
        }
        else {
            param[PARAM_SUCCESS].i = 1;
        }

        free(Aref); free(work);
    }

    //================================================================
    // Free arrays
    //================================================================
    free(A); free(B); free(X);
}
