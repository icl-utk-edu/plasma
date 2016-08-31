/**
 *
 * @file test_zgbtrf.c
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

/***************************************************************************//**
 *
 * @brief Tests ZGBTRF.
 *
 * @param[in]  param - array of parameters
 * @param[out] info  - string of column labels or column values; length InfoLen
 *
 * If param is NULL and info is NULL,     print usage and return.
 * If param is NULL and info is non-NULL, set info to column labels and return.
 * If param is non-NULL and info is non-NULL, set info to column values
 * and run test.
 ******************************************************************************/
void test_zgbtrf(param_value_t param[], char *info)
{
    //================================================================
    // Print usage info or return column labels or values.
    //================================================================
    if (param == NULL) {
        if (info == NULL) {
            // Print usage info.
            //  gbtrf params
            print_usage(PARAM_M);
            print_usage(PARAM_N);
            print_usage(PARAM_KL);
            print_usage(PARAM_KU);
            print_usage(PARAM_PADA);
            print_usage(PARAM_NB);
            //  gbtrs params for check
            print_usage(PARAM_NRHS);
            print_usage(PARAM_PADB);
        }
        else {
            // Return column labels.
            snprintf(info, InfoLen,
                "%*s %*s %*s %*s %*s %*s %*s %*s ",
                InfoSpacing, "M",
                InfoSpacing, "N",
                InfoSpacing, "KL",
                InfoSpacing, "KU",
                InfoSpacing, "PadA",
                InfoSpacing, "NB",
                InfoSpacing, "NRHS",
                InfoSpacing, "PadB");
        }
        return;
    }
    // Return column values.
    snprintf(info, InfoLen,
        "%*d %*d %*d %*d %*d %*d %*d %*d ",
        InfoSpacing, param[PARAM_M].i,
        InfoSpacing, param[PARAM_N].i,
        InfoSpacing, param[PARAM_KL].i,
        InfoSpacing, param[PARAM_KU].i,
        InfoSpacing, param[PARAM_PADA].i,
        InfoSpacing, param[PARAM_NB].i,
        InfoSpacing, param[PARAM_NRHS].i,
        InfoSpacing, param[PARAM_PADB].i);

    //================================================================
    // Set parameters.
    //================================================================
    int pada = param[PARAM_PADA].i;
    int m    = param[PARAM_M].i;
    int n    = param[PARAM_N].i;
    int nb   = param[PARAM_NB].i;
    int lda  = imax(1, m + pada);

    int kl   = param[PARAM_KL].i;
    int ku   = param[PARAM_KU].i;
    int kut  = (ku+kl+nb-1)/nb; // number of tiles in upper band (not including diagonal)
    int klt  = (kl+nb-1)/nb;    // number of tiles in lower band (not including diagonal)
    int ldab = (kut+klt+1)*nb;  // since we use zgetrf on panel, we pivot back within panel.
                                // this could fill the last tile of the panel,
                                // and we need extra NB space on the bottom

    int test = param[PARAM_TEST].c == 'y';
    double tol = param[PARAM_TOL].d * LAPACKE_dlamch('E');

    //================================================================
    // Set tuning parameters.
    //================================================================
    PLASMA_Set(PLASMA_TILE_SIZE, param[PARAM_NB].i);

    //================================================================
    // Allocate and initialize arrays.
    //================================================================
    /* band matrix A in full storage (also used for solution check) */
    PLASMA_Complex64_t *A =
        (PLASMA_Complex64_t*)malloc((size_t)lda*n*sizeof(PLASMA_Complex64_t));
    assert(A != NULL);

    int seed[] = {0, 0, 0, 1};
    lapack_int retval;
    retval = LAPACKE_zlarnv(1, seed, (size_t)lda*n, A);
    assert(retval == 0);

    int i,j;
    /* zero out elements outside the band */
    PLASMA_Complex64_t zzero = 0.0;
    for (i=0; i<m; i++) {
        for (j=i+ku+1; j<n; j++) A[i + j*lda] = zzero;
    }
    for (j=0; j<n; j++) {
        for (i=j+kl+1; i<m; i++) A[i + j*lda] = zzero;
    }

    /* band matrix A in skewed LAPACK storage */
    PLASMA_Complex64_t *AB = NULL;
    AB = (PLASMA_Complex64_t*)malloc((size_t)ldab*n*sizeof(PLASMA_Complex64_t));
    assert(AB != NULL);

    /* convert into LAPACK's skewed storage */
    for (j=0; j<n; j++) {
        int i_kl = imax(0,   j-ku);
        int i_ku = imin(m-1, j+kl);
        for (i=0; i<ldab; i++) AB[i + j*ldab] = zzero;
        for (i=i_kl; i<=i_ku; i++) AB[kl + i-(j-ku) + j*ldab] = A[i + j*lda];
    }

    /* pivot indices */
    int *ipiv = NULL, *fill = NULL;
    ipiv = (int*)malloc((2*imin(m, n))*sizeof(int));
    assert(ipiv != NULL);
    fill = &ipiv[imin(m,n)];

    //================================================================
    // Run and time PLASMA.
    //================================================================
    int iinfo;

    plasma_time_t start = omp_get_wtime();
    iinfo = PLASMA_zgbtrf(m, n, kl, ku, AB, ldab, ipiv, fill);
    if (iinfo != 0) printf( " zgbtrf failed with info=%d\n",iinfo );
    plasma_time_t stop = omp_get_wtime();
    plasma_time_t time = stop-start;

    param[PARAM_TIME].d = time;
    param[PARAM_GFLOPS].d = 0.0 / time / 1e9;

    //================================================================
    // Test results by computing residual norm.
    //================================================================
    if (test && n == m) {
        PLASMA_Complex64_t zone =   1.0;
        PLASMA_Complex64_t zmone = -1.0;

        int nrhs = param[PARAM_NRHS].i;
        int ldb = imax(1, n + param[PARAM_PADB].i);

        /* set up right-hand-side B */
        PLASMA_Complex64_t *B = NULL;
        B = (PLASMA_Complex64_t*)malloc((size_t)ldb*nrhs*sizeof(PLASMA_Complex64_t));
        assert(B != NULL);

        retval = LAPACKE_zlarnv(1, seed, (size_t)ldb*nrhs, B);
        assert(retval == 0);
    
        /* copy B to X */
        int ldx = ldb;
        PLASMA_Complex64_t *X = NULL;
        X = (PLASMA_Complex64_t*)malloc((size_t)ldx*nrhs*sizeof(PLASMA_Complex64_t));
        assert(X != NULL);
        LAPACKE_zlacpy_work(LAPACK_COL_MAJOR,'F', m, nrhs, B, ldb, X, ldx);

        /* solve for X */
        iinfo = PLASMA_zgbtrs(PlasmaNoTrans, n, kl, ku, nrhs,
                              AB, ldab, ipiv, X, ldb);
        if (iinfo != 0) printf( " zgbtrs failed with info = %d\n",iinfo );

        /* compute residual vector */
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, nrhs, n,
                    CBLAS_SADDR(zmone), A, lda,
                                        X, ldx,
                    CBLAS_SADDR(zone),  B, ldb);

        /* compute various norms */
        double *work = NULL;
        work = (double*)malloc((size_t)m*sizeof(double));
        assert(work != NULL);

        double Anorm = LAPACKE_zlange_work(LAPACK_COL_MAJOR, 'F', m, n,    A, lda, work);
        double Xnorm = LAPACKE_zlange_work(LAPACK_COL_MAJOR, 'I', n, nrhs, X, ldb, work);
        double Rnorm = LAPACKE_zlange_work(LAPACK_COL_MAJOR, 'I', n, nrhs, B, ldb, work);
        double residual = Rnorm/(n*Anorm*Xnorm);

        param[PARAM_ERROR].d = residual;
        param[PARAM_SUCCESS].i = residual < tol;

        /* free arrays */
        free(work);
        free(X);
        free(B);
    }

    //================================================================
    // Free arrays.
    //================================================================
    free(ipiv);
    free(AB);
    free(A);
}
