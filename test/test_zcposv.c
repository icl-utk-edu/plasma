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

#include "core_blas.h"
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

#include "plasma_types.h"
#include "plasma_async.h"
#include "plasma_context.h"
#include "plasma_descriptor.h"
#include "plasma_z.h"

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
            print_usage(PARAM_PADX);
        }
        else {
            // Return column labels
            snprintf(info, InfoLen,
                "%*s %*s %*s %*s %*s %*s",
                InfoSpacing, "UpLo",
                InfoSpacing, "n",
                InfoSpacing, "nrhs",
                InfoSpacing, "PadA",
                InfoSpacing, "PadB",
                InfoSpacing, "PadX");
        }
        return;
    }
    // Return column values
    snprintf(info, InfoLen,
        "%*c %*d %*d %*d %*d %*d",
        InfoSpacing, param[PARAM_UPLO].c,
        InfoSpacing, param[PARAM_N].i,
        InfoSpacing, param[PARAM_NRHS].i,
        InfoSpacing, param[PARAM_PADA].i,
        InfoSpacing, param[PARAM_PADB].i,
        InfoSpacing, param[PARAM_PADX].i);

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
    int ldx  = imax(1, n + param[PARAM_PADX].i);
    int ITER;

    double eps = LAPACKE_dlamch_work('e');

    //================================================================
    // Allocate and initialize arrays
    //================================================================
    /*
    PLASMA_Complex64_t *A1   = (PLASMA_Complex64_t *)malloc(LDA*N   *sizeof(PLASMA_Complex64_t));
    PLASMA_Complex64_t *A2   = (PLASMA_Complex64_t *)malloc(LDA*N   *sizeof(PLASMA_Complex64_t));
    PLASMA_Complex64_t *B1   = (PLASMA_Complex64_t *)malloc(LDB*NRHS*sizeof(PLASMA_Complex64_t));
    PLASMA_Complex64_t *B2   = (PLASMA_Complex64_t *)malloc(LDB*NRHS*sizeof(PLASMA_Complex64_t));
    */

    PLASMA_Complex64_t *A0 =
        (PLASMA_Complex64_t *)malloc((size_t)lda*n*sizeof(PLASMA_Complex64_t));
    assert(A0 != NULL);

    PLASMA_Complex64_t *A =
        (PLASMA_Complex64_t *)malloc((size_t)lda*n*sizeof(PLASMA_Complex64_t));
    assert(A != NULL);

    PLASMA_Complex64_t *B =
        (PLASMA_Complex64_t *)malloc((size_t)ldb*nrhs*sizeof(PLASMA_Complex64_t));
    assert(B != NULL);

    PLASMA_Complex64_t *X =
        (PLASMA_Complex64_t *)malloc((size_t)ldx*nrhs*sizeof(PLASMA_Complex64_t));
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

    /* Initialise A(lda x n) as Symmetric Positive Definite matrix
     *(Hessenberg in case of complex precision)
     * Generate random matrix A0 */
    int seed[] = {0, 0, 0, 1};
    lapack_int retval = LAPACKE_zlarnv(1, seed, (size_t)lda*n, A0);
    assert(retval == 0);

    // Make A symmetric: A = A0*A0'
    double alpha = 1.0, beta = 1.0;

    retval = cblas_zgemm(CblasColMajor, CblasNoTrans, CblasTrans, lda, n, n,
                         alpha, A0, lda, A0, lda, beta, A, lda);
    assert(retval == 0);

    // Release memory for auxiliary matrix A0
    free(A0);

    // Ensure diagonal dominance: A = A + n*eye(n)
    for (int i = 0; i < lda; i++) {
        A[i*n+i] *= lda;
    }

    // Initialise random matrix B(ldb x nrhs)
    lapack_int retval = LAPACKE_zlarnv(1, seed, (size_t)ldb*nrhs, B);
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
    // param[PARAM_GFLOPS].d = flops_zcposv(side, n, nrhs) / time / 1e9;
    param[PARAM_GFLOPS].d = 0.0;



    /*-------------------------------------------------------------
    *  TESTING ZCPOSV
    */

    printf("\n");
    printf("------ TESTS FOR PLASMA ZCPOSV ROUTINE ------  \n");
    printf("            Size of the Matrix %d by %d\n", N, N);
    printf("\n");
    printf(" The matrix A is randomly generated for each test.\n");
    printf("============\n");
    printf(" The relative machine precision (eps) is to be %e \n", eps);
    printf(" Computational tests pass if scaled residuals are less than 60.\n");

    /* PLASMA ZCPOSV */
    uplo = PlasmaLower;
    info = PLASMA_zcposv(uplo, N, NRHS, A2, LDA, B1, LDB, B2, LDB, &ITER);

    if (info != PLASMA_SUCCESS ) {
        printf("PLASMA_zcposv is not completed: info = %d\n", info);
        info_solution = 1;
    } else {
        printf(" Solution obtained with %d iterations\n", ITER);

        /* Check the factorization and the solution */
        info_solution = check_solution(N, NRHS, A1, LDA, B1, B2, LDB, eps);
    }
    
    if (info_solution == 0){
        printf("***************************************************\n");
        printf(" ---- TESTING ZCPOSV ..................... PASSED !\n");
        printf("***************************************************\n");
    }
    else{
        printf("***************************************************\n");
        printf(" - TESTING ZCPOSV .. FAILED !\n");
        printf("***************************************************\n");
    }

    free(A1); free(A2); free(B1); free(B2);
    
    return 0;
}

/*------------------------------------------------------------------------
 *  Check the accuracy of the solution of the linear system
 */

static int check_solution(int N, int NRHS, PLASMA_Complex64_t *A1, int LDA, PLASMA_Complex64_t *B1, PLASMA_Complex64_t *B2, int LDB, double eps )
{
    int info_solution;
    double Rnorm, Anorm, Xnorm, Bnorm, result;
    PLASMA_Complex64_t alpha, beta;
    double *work = (double *)malloc(N*sizeof(double));

    alpha = 1.0;
    beta  = -1.0;

    Xnorm = LAPACKE_zlange_work(LAPACK_COL_MAJOR, lapack_const(PlasmaInfNorm), N, NRHS, B2, LDB, work);
    Anorm = LAPACKE_zlange_work(LAPACK_COL_MAJOR, lapack_const(PlasmaInfNorm), N, N, A1, LDA, work);
    Bnorm = LAPACKE_zlange_work(LAPACK_COL_MAJOR, lapack_const(PlasmaInfNorm), N, NRHS, B1, LDB, work);

    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, N, NRHS, N, CBLAS_SADDR(alpha), A1, LDA, B2, LDB, CBLAS_SADDR(beta), B1, LDB);
    Rnorm = LAPACKE_zlange_work(LAPACK_COL_MAJOR, lapack_const(PlasmaInfNorm), N, NRHS, B1, LDB, work);

    if (getenv("PLASMA_TESTING_VERBOSE"))
      printf( "||A||_oo=%f\n||X||_oo=%f\n||B||_oo=%f\n||A X - B||_oo=%e\n", Anorm, Xnorm, Bnorm, Rnorm );

    result = Rnorm / ( (Anorm*Xnorm+Bnorm)*N*eps ) ;
    printf("============\n");
    printf("Checking the Residual of the solution \n");
    printf("-- ||Ax-B||_oo/((||A||_oo||x||_oo+||B||_oo).N.eps) = %e \n", result);

    if (  isnan(Xnorm) || isinf(Xnorm) || isnan(result) || isinf(result) || (result > 60.0) ) {
        printf("-- The solution is suspicious ! \n");
        info_solution = 1;
     }
    else{
        printf("-- The solution is CORRECT ! \n");
        info_solution = 0;
    }

    free(work);

    return info_solution;
}
