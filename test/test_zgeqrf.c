/**
 *
 * @file test_zgeqrf.c
 *
 *  PLASMA test routine.
 *  PLASMA is a software package provided by Univ. of Tennessee,
 *  Univ. of California Berkeley, Univ. of Colorado Denver and
 *  Univ. of Manchester.
 *
 * @version 3.0.0
 * @author Jakub Sistek
 * @date 2016-7-9
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

#include "plasma_types.h"
#include "plasma_async.h"
#include "plasma_context.h"
#include "plasma_descriptor.h"
#include "plasma_z.h"

#define COMPLEX

/***************************************************************************//**
 *
 * @brief Tests ZGEQRF.
 *
 * @param[in]  param - array of parameters
 * @param[out] info  - string of column labels or column values; length InfoLen
 *
 * If param is NULL     and info is NULL,     print usage and return.
 * If param is NULL     and info is non-NULL, set info to column headings and return.
 * If param is non-NULL and info is non-NULL, set info to column values   and run test.
 ******************************************************************************/
void test_zgeqrf(param_value_t param[], char *info)
{
    //================================================================
    // Print usage info or return column labels or values.
    //================================================================
    if (param == NULL) {
        if (info == NULL) {
            // Print usage info.
            print_usage(PARAM_M);
            print_usage(PARAM_N);
            print_usage(PARAM_PADA);
        }
        else {
            // Return column labels.
            snprintf(info, InfoLen,
                "%*s %*s %*s",
                InfoSpacing, "M",
                InfoSpacing, "N",
                InfoSpacing, "PadA");
        }
        return;
    }
    // Return column values.
    snprintf(info, InfoLen,
        "%*d %*d %*d",
        InfoSpacing, param[PARAM_M].i,
        InfoSpacing, param[PARAM_N].i,
        InfoSpacing, param[PARAM_PADA].i);

    //================================================================
    // Set parameters.
    //================================================================
    int m = param[PARAM_M].i;
    int n = param[PARAM_N].i;

    int lda = imax(1, m + param[PARAM_PADA].i);

    int test = param[PARAM_TEST].c == 'y';
    //double tol = param[PARAM_TOL].d * LAPACKE_dlamch('E');
    double tol = param[PARAM_TOL].d;
    double eps = LAPACKE_dlamch('E');

    //================================================================
    // Allocate and initialize arrays.
    //================================================================
    PLASMA_Complex64_t *A =
        (PLASMA_Complex64_t*)malloc((size_t)lda*n*sizeof(PLASMA_Complex64_t));
    assert(A != NULL);

    int seed[] = {0, 0, 0, 1};
    lapack_int retval;
    retval = LAPACKE_zlarnv(1, seed, (size_t)lda*n, A);
    assert(retval == 0);

    PLASMA_Complex64_t *Aref = NULL;
    if (test) {
        Aref = (PLASMA_Complex64_t*)malloc(
            (size_t)lda*n*sizeof(PLASMA_Complex64_t));
        assert(Aref != NULL);

        memcpy(Aref, A, (size_t)lda*n*sizeof(PLASMA_Complex64_t));
    }

    // Get PLASMA context.
    plasma_context_t *plasma = plasma_context_self();
    // Initialize tile matrix descriptor for matrix T
    // using multiples of tile size.
    int nb = plasma->nb;
    int ib = nb;
    int mt = (m%nb == 0) ? (m/nb) : (m/nb+1);
    int nt = (n%nb == 0) ? (n/nb) : (n/nb+1);
    PLASMA_desc descT = plasma_desc_init(PlasmaComplexDouble, ib, nb, ib*nb,
                                         mt*ib, nt*nb, 0, 0, mt*ib, nt*nb);
    // allocate memory for the matrix T
    retval = plasma_desc_mat_alloc(&descT);
    assert(retval == 0);

    //================================================================
    // Run and time PLASMA.
    //================================================================
    plasma_time_t start = omp_get_wtime();
    PLASMA_zgeqrf(m, n, A, lda, &descT);
    plasma_time_t stop = omp_get_wtime();
    plasma_time_t time = stop-start;

    param[PARAM_TIME].d = time;
    param[PARAM_GFLOPS].d = flops_zgeqrf(m,n) / time / 1e9;

    //================================================================
    // Test results by solving a linear system.
    //================================================================
    if (test) {
        const int nrhs = 1;
        const int ldb  = m;

        // |A|_F
        double work[1];
        double Anorm = LAPACKE_zlange_work(LAPACK_COL_MAJOR, 'F', m, n,
                                           Aref, lda, work);

        // prepare right-hand side B, store into initial X
        PLASMA_Complex64_t *X = NULL;
        X = (PLASMA_Complex64_t*)malloc((size_t)ldb*nrhs*
                                        sizeof(PLASMA_Complex64_t));
        assert(X != NULL);

        PLASMA_Complex64_t *B = NULL;
        B = (PLASMA_Complex64_t*)malloc((size_t)ldb*nrhs*
                                        sizeof(PLASMA_Complex64_t));
        assert(B != NULL);

        retval = LAPACKE_zlarnv(1, seed, (size_t)m*nrhs, B);
        assert(retval == 0);
        memcpy(X, B, (size_t)ldb*nrhs*sizeof(PLASMA_Complex64_t));

        // |B|_F
        double Bnorm = LAPACKE_zlange_work(LAPACK_COL_MAJOR, 'F', m, nrhs,
                                           B, ldb, work);

        // Call PLASMA function for solving R*X = Q'*B
        PLASMA_zgeqrs(m, n, nrhs, A, lda, &descT, X, ldb);

        // |X|_F
        double Xnorm = LAPACKE_zlange_work(LAPACK_COL_MAJOR, 'F', n, nrhs,
                                           X, ldb, work);

        // compute residual and store it in B = A*X - B
        PLASMA_Complex64_t zone  =  1.0;
        PLASMA_Complex64_t mzone = -1.0;
        PLASMA_Complex64_t zzero =  0.0;
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, nrhs, n,
                    CBLAS_SADDR(zone), Aref, lda, X, ldb,
                    CBLAS_SADDR(mzone), B, ldb);

        // Compute A' * (Ax - b)
        cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, n, nrhs, m,
                    CBLAS_SADDR(zone), Aref, lda, B, ldb,
                    CBLAS_SADDR(zzero), X, ldb);

        // |RES|_F
        double Rnorm = LAPACKE_zlange_work(LAPACK_COL_MAJOR, 'F', n, nrhs,
                                           X, ldb, work);

        // normalize the result
        double result = Rnorm / ( (Anorm*Xnorm+Bnorm)*n*eps);

        param[PARAM_ERROR].d = result;
        param[PARAM_SUCCESS].i = result < tol;

        free(B);
        free(X);
    }

    //================================================================
    // Free arrays.
    //================================================================
    plasma_desc_mat_free(&descT);
    free(A);
    if (test)
        free(Aref);
}
