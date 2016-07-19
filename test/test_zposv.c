/**
 *
 * @file test_zposv.c
 *
 *  PLASMA test routine.
 *  PLASMA is a software package provided by Univ. of Tennessee,
 *  Univ. of California Berkeley, Univ. of Colorado Denver and
 *  Univ. of Manchester.
 *
 * @version 
 * @author Mawussi Zounon
 * @date 
 * @precisions normal z -> s d c
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
#include <plasma.h>

#define COMPLEX

#define A(i,j)  A[i + j*lda]

/***************************************************************************//**
 *
 * @brief Tests ZPOSV.
 *
 * @param[in]  param - array of parameters
 * @param[out] info  - string of column labels or column values; length InfoLen
 *
 * If param is NULL     and info is NULL,     print usage and return.
 * If param is NULL     and info is non-NULL, set info to column headings
 *  and return.
 * If param is non-NULL and info is non-NULL, set info to column values
 *  and run test.
 ******************************************************************************/
void test_zposv(param_value_t param[], char *info)
{
    //================================================================
    // Print usage info or return column labels or values.
    //================================================================
    if (param == NULL) {
        if (info == NULL) {
            // Print usage info.
            print_usage(PARAM_UPLO);
            print_usage(PARAM_M);
	    print_usage(PARAM_NRHS);
            print_usage(PARAM_PADA);
	    print_usage(PARAM_PADB);
        }
        else {
	  // Return column labels.
	  snprintf(info, InfoLen,
		   "%*s %*s %*s %*s %*s",
		   InfoSpacing, "Uplo",
		   InfoSpacing, "N",
		   InfoSpacing, "NRHS",
		   InfoSpacing, "PadA",
		   InfoSpacing, "PadB");
        }
        return;
    }
    // Return column values.
    snprintf(info, InfoLen,
	     "%*c %*d %*d %*d %*d",
	     InfoSpacing, param[PARAM_UPLO].c,
	     InfoSpacing, param[PARAM_N].i,
	     InfoSpacing, param[PARAM_NRHS].i,
	     InfoSpacing, param[PARAM_PADA].i,
	     InfoSpacing, param[PARAM_PADB].i);
    
    //================================================================
    // Set parameters.
    //================================================================
    PLASMA_enum uplo;

    if (param[PARAM_UPLO].c == 'l')
        uplo = PlasmaLower;
    else 
        uplo = PlasmaUpper;
    
    int n = param[PARAM_N].i;
    int nrhs = param[PARAM_NRHS].i;

    int lda = imax(1, n + param[PARAM_PADA].i);
    int ldb = imax(1, n + param[PARAM_PADB].i);
    
    int test = param[PARAM_TEST].c == 'y';
    double tol = param[PARAM_TOL].d * LAPACKE_dlamch('E');

    //================================================================
    // Allocate and initialize arrays.
    //================================================================
    PLASMA_Complex64_t *A =
        (PLASMA_Complex64_t*)malloc((size_t)lda*n*sizeof(PLASMA_Complex64_t));
    assert(A != NULL);

    PLASMA_Complex64_t *B =
        (PLASMA_Complex64_t*)malloc((size_t)ldb*nrhs*sizeof(PLASMA_Complex64_t));
    assert(B != NULL);
    
    int seed[] = {0, 0, 0, 1};
    lapack_int retval;
    retval = LAPACKE_zlarnv(1, seed, (size_t)lda*n, A);
    assert(retval == 0);
    retval = LAPACKE_zlarnv(1, seed, (size_t)ldb*nrhs, B);
    assert(retval == 0);

    //================================================================
    // Make the A matrix symmetric/Hermitian positive definite.
    // It increases diagonal by n, and makes it real.
    // It sets Aji = conj( Aij ) for j < i, that is, copy lower 
    // triangle to upper triangle.
    //================================================================

    for(int i = 0; i < n; ++i ) {
        A(i,i) = (creal(A(i,i)) + n) + 0. * I;
        for(int j = 0; j < i; ++j ) {
            A(j,i) = conj(A(i,j));
        }
    }

    PLASMA_Complex64_t *Aref = NULL;
    PLASMA_Complex64_t *Bref = NULL;
    if (test) {
        Aref = (PLASMA_Complex64_t*)malloc(
            (size_t)lda*n*sizeof(PLASMA_Complex64_t));
        assert(Aref != NULL);

	Bref = (PLASMA_Complex64_t*)malloc(
            (size_t)ldb*nrhs*sizeof(PLASMA_Complex64_t));
        assert(Bref != NULL);

	memcpy(Aref, A, (size_t)lda*n*sizeof(PLASMA_Complex64_t));
        memcpy(Bref, B, (size_t)ldb*nrhs*sizeof(PLASMA_Complex64_t));
    }

    //================================================================
    // Run and time PLASMA.
    //================================================================
    plasma_time_t start = omp_get_wtime();
    PLASMA_zposv((CBLAS_UPLO)uplo, n, nrhs, A, lda, B, ldb);
    plasma_time_t stop = omp_get_wtime();
    plasma_time_t time = stop-start;
    double flops = flops_zpotrf(n) + 2*flops_ztrsm(uplo, n, nrhs);
    param[PARAM_TIME].d = time;    
    param[PARAM_GFLOPS].d = flops / time / 1e9;

    //================================================================
    // Test results by comparing to a reference implementation.
    //================================================================
    if (test) {

        LAPACKE_zposv(
            LAPACK_COL_MAJOR,
            lapack_const(uplo), n, nrhs,
            Aref, lda,
	    Bref, ldb);
        
        PLASMA_Complex64_t zmone = -1.0;
        cblas_zaxpy((size_t)ldb*nrhs, CBLAS_SADDR(zmone), Bref, 1, B, 1);

        double work[1];
        double Bnorm = LAPACKE_zlange_work(
                           LAPACK_COL_MAJOR, 'F', n, nrhs, Bref, ldb, work);
        double error = LAPACKE_zlange_work(
                           LAPACK_COL_MAJOR, 'F', n, nrhs, B,    ldb, work);
        if (Bnorm != 0)
            error /= Bnorm;

        param[PARAM_ERROR].d = error;
        param[PARAM_SUCCESS].i = error < tol;
    }

    //================================================================
    // Free arrays.
    //================================================================
    free(A);
    free(B);
      if (test){
        free(Aref);
	free(Bref);
      }
}
