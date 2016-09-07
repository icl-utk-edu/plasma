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
#include "core_lapack.h"
#include "plasma.h"

#include <assert.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <omp.h>

#define COMPLEX

/***************************************************************************//**
 *
 * @brief Tests ZGELS.
 *
 * @param[in]  param - array of parameters
 * @param[out] info  - string of column labels or column values; length InfoLen
 *
 * If param is NULL and info is NULL,     print usage and return.
 * If param is NULL and info is non-NULL, set info to column labels and return.
 * If param is non-NULL and info is non-NULL, set info to column values
 * and run test.
 ******************************************************************************/
void test_zgels(param_value_t param[], char *info)
{
    //================================================================
    // Print usage info or return column labels or values.
    //================================================================
    if (param == NULL) {
        if (info == NULL) {
            // Print usage info.
            print_usage(PARAM_M);
            print_usage(PARAM_N);
            print_usage(PARAM_NRHS);
            print_usage(PARAM_PADA);
            print_usage(PARAM_PADB);
            print_usage(PARAM_NB);
            print_usage(PARAM_IB);
        }
        else {
            // Return column labels.
            snprintf(info, InfoLen,
                     "%*s %*s %*s %*s %*s %*s %*s",
                     InfoSpacing, "M",
                     InfoSpacing, "N",
                     InfoSpacing, "NRHS",
                     InfoSpacing, "PadA",
                     InfoSpacing, "PadB",
                     InfoSpacing, "NB",
                     InfoSpacing, "IB");
        }
        return;
    }
    // Return column values.
    snprintf(info, InfoLen,
             "%*d %*d %*d %*d %*d %*d %*d",
             InfoSpacing, param[PARAM_M].i,
             InfoSpacing, param[PARAM_N].i,
             InfoSpacing, param[PARAM_NRHS].i,
             InfoSpacing, param[PARAM_PADA].i,
             InfoSpacing, param[PARAM_PADB].i,
             InfoSpacing, param[PARAM_NB].i,
             InfoSpacing, param[PARAM_IB].i);

    //================================================================
    // Set parameters.
    //================================================================
    int m    = param[PARAM_M].i;
    int n    = param[PARAM_N].i;
    int nrhs = param[PARAM_NRHS].i;

    int lda = imax(1, m + param[PARAM_PADA].i);
    int ldb = imax(1, imax(m, n) + param[PARAM_PADB].i);

    int test = param[PARAM_TEST].c == 'y';
    double tol = param[PARAM_TOL].d * LAPACKE_dlamch('E');

    //================================================================
    // Set tuning parameters.
    //================================================================
    PLASMA_Set(PLASMA_TILE_SIZE,        param[PARAM_NB].i);
    PLASMA_Set(PLASMA_INNER_BLOCK_SIZE, param[PARAM_IB].i);

    //================================================================
    // Allocate and initialize arrays.
    //================================================================
    PLASMA_Complex64_t *A =
        (PLASMA_Complex64_t*)malloc((size_t)lda*n*sizeof(PLASMA_Complex64_t));
    assert(A != NULL);

    PLASMA_Complex64_t *B =
        (PLASMA_Complex64_t*)malloc((size_t)ldb*nrhs*
                                    sizeof(PLASMA_Complex64_t));
    assert(B != NULL);

    int seed[] = {0, 0, 0, 1};
    lapack_int retval;
    retval = LAPACKE_zlarnv(1, seed, (size_t)lda*n, A);
    assert(retval == 0);

    retval = LAPACKE_zlarnv(1, seed, (size_t)ldb*nrhs, B);
    assert(retval == 0);

    // store the original arrays if residual is to be evaluated
    PLASMA_Complex64_t *Aref = NULL;
    PLASMA_Complex64_t *Bref = NULL;
    if (test) {
        Aref = (PLASMA_Complex64_t*)malloc(
            (size_t)lda*n*sizeof(PLASMA_Complex64_t));
        assert(Aref != NULL);

        memcpy(Aref, A, (size_t)lda*n*sizeof(PLASMA_Complex64_t));

        Bref = (PLASMA_Complex64_t*)malloc(
            (size_t)ldb*nrhs*sizeof(PLASMA_Complex64_t));
        assert(Bref != NULL);

        memcpy(Bref, B, (size_t)ldb*nrhs*sizeof(PLASMA_Complex64_t));
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
    PLASMA_zgels(PlasmaNoTrans, m, n, nrhs,
                 A, lda,
                 &descT,
                 B, ldb);
    plasma_time_t stop = omp_get_wtime();
    plasma_time_t time = stop-start;

    param[PARAM_TIME].d = time;

    double flop;
    if (m >= n) {
        // cost of QR-based factorization and solve
        flop = flops_zgeqrf(m,n) + flops_zgeqrs(m,n,nrhs);
    }
    else {
        // cost of LQ-based factorization, triangular solve, and Q^H application
        flop = flops_zgelqf(m,n) +
               flops_ztrsm(PlasmaLeft,m,nrhs) +
               flops_zunmlq(PlasmaLeft,n,nrhs,m);
    }
    param[PARAM_GFLOPS].d = flop / time / 1e9;

    //================================================================
    // Test results by solving a linear system.
    //================================================================
    if (test) {
        // |A|_F
        double work[1];
        double Anorm = LAPACKE_zlange_work(LAPACK_COL_MAJOR, 'F', m, n,
                                           Aref, lda, work);

        // |B|_F
        double Bnorm = LAPACKE_zlange_work(LAPACK_COL_MAJOR, 'F', m, nrhs,
                                           Bref, ldb, work);

        // |X|_F, solution X is now stored in the n-by-nrhs part of B
        double Xnorm = LAPACKE_zlange_work(LAPACK_COL_MAJOR, 'F', n, nrhs,
                                           B, ldb, work);

        // compute residual and store it in B = A*X - B
        PLASMA_Complex64_t zone  =  1.0;
        PLASMA_Complex64_t mzone = -1.0;
        PLASMA_Complex64_t zzero =  0.0;
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, nrhs, n,
                    CBLAS_SADDR(zone), Aref, lda, B, ldb,
                    CBLAS_SADDR(mzone), Bref, ldb);

        // Compute B = A^H * (A*X - B)
        cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, n, nrhs, m,
                    CBLAS_SADDR(zone), Aref, lda, Bref, ldb,
                    CBLAS_SADDR(zzero), B, ldb);

        // |RES|_F
        double Rnorm = LAPACKE_zlange_work(LAPACK_COL_MAJOR, 'F', n, nrhs,
                                           B, ldb, work);

        // normalize the result
        double result = Rnorm / ((Anorm*Xnorm+Bnorm)*n);

        param[PARAM_ERROR].d = result;
        param[PARAM_SUCCESS].i = result < tol;
    }

    //================================================================
    // Free arrays.
    //================================================================
    plasma_desc_mat_free(&descT);
    free(A);
    free(B);
    if (test) {
        free(Aref);
        free(Bref);
    }
}
