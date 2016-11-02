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
#include "core_blas.h"
#include "core_lapack.h"
#include "plasma.h"

#include <assert.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <omp.h>

#define A(m, n) (plasma_complex64_t*)plasma_tile_addr(A, m, n)

void pzdesc2ge(plasma_desc_t A, plasma_complex64_t *pA, int lda);
void pzge2desc(plasma_complex64_t *pA, int lda, plasma_desc_t A);

/******************************************************************************/
void plasma_zgetrf_(int m, int n,
                    plasma_complex64_t *pA, int lda, int *ipiv,
                    int nb, int ib)
{
    plasma_desc_t A;
    int retval = plasma_desc_general_create(PlasmaComplexDouble, nb, nb,
                                            m, n, 0, 0, m, n, &A);
    assert(retval == PlasmaSuccess);


    double sfmin = LAPACKE_dlamch_work('S');


    for (int k = 0; k < imin(m, n); k += ib) {

        int kb = imin(imin(m, n)-k, ib);

        // panel factorization
        for (int j = k; j < k+kb; j++) {

            // pivot search
            int imax = 0;
            plasma_complex64_t max = pA[j+j*lda];
            for (int i = 1; i < m-j; i++)
                if (cblas_dcabs1(&pA[j+i+j*lda]) > cblas_dcabs1(&max)) {
                    max = pA[j+i+j*lda];
                    imax = i;
                }
            int jp = j+imax;
            ipiv[j] = jp-k+1;

pzge2desc(pA, lda, A);

            plasma_complex64_t *a0 = A(0, 0);
            plasma_complex64_t *ap = A(jp/nb, 0);

            int lda0 = plasma_tile_mmain(A, 0);
            int ldap = plasma_tile_mmain(A, jp/nb);

            // pivot swap
            cblas_zswap(kb,
                        &a0[j+k*lda0], lda0,
                        &ap[jp%nb+k*ldap], ldap);

            // column scaling and trailing update
            for (int l = 0; l < A.mt; l++) {

                plasma_complex64_t *al = A(l, 0);
                int ldal = plasma_tile_mmain(A, l);

                int mva0 = plasma_tile_mview(A, 0);
                int mval = plasma_tile_mview(A, l);

                // column scaling
                if (cabs(a0[j+j*lda0]) >= sfmin) {
                    if (l == 0) {
                        for (int i = 1; i < mva0-j; i++)
                            a0[j+i+j*lda0] /= a0[j+j*lda0];
                    }
                    else {
                        for (int i = 0; i < mval; i++)
                            al[i+j*ldal] /= a0[j+j*lda0];
                    }
                }
                else {
                    plasma_complex64_t scal = 1.0/a0[j+j*lda0];
                    if (l == 0)
                        cblas_zscal(mva0-j-1, CBLAS_SADDR(scal),
                                    &a0[j+1+j*lda0], 1);
                    else
                        cblas_zscal(mval, CBLAS_SADDR(scal), &al[j*ldal], 1);
                }

                // trailing update
                plasma_complex64_t zmone = -1.0;
                if (l == 0) {
                    cblas_zgeru(CblasColMajor,
                                mva0-j-1, k+kb-j-1,
                                CBLAS_SADDR(zmone), &a0[j+1+j*lda0], 1,
                                                    &a0[j+(j+1)*lda0], lda0,
                                                    &a0[j+1+(j+1)*lda0], lda0);
                }
                else {
                    cblas_zgeru(CblasColMajor,
                                mval, k+kb-j-1,
                                CBLAS_SADDR(zmone), &al[+j*ldal], 1,
                                                    &a0[j+(j+1)*lda0], lda0,
                                                    &al[+(j+1)*ldal], ldal);
                }

            }

pzdesc2ge(A, pA, lda);

        }

pzge2desc(pA, lda, A);

        // pivot adjustment
        for (int i = k+1; i <= imin(m, k+kb); i++)
            ipiv[i-1] += k;

        plasma_complex64_t *a0 = A(0, 0);

        int lda0 = plasma_tile_mmain(A, 0);
        int nva0 = plasma_tile_nview(A, 0);
        int mva0 = plasma_tile_mview(A, 0);

        // right pivoting
        for (int i = k; i < k+kb; i++) {

            plasma_complex64_t *ap = A((ipiv[i]-1)/nb, 0);
            int ldap = plasma_tile_mmain(A, (ipiv[i]-1)/nb);

            cblas_zswap(nva0-k-kb,
                        &a0[i+(k+kb)*lda0], lda0,
                        &ap[(ipiv[i]-1)%nb+(k+kb)*ldap], ldap);
        }

        // trsm
        plasma_complex64_t zone = 1.0;
        cblas_ztrsm(CblasColMajor,
                    PlasmaLeft, PlasmaLower,
                    PlasmaNoTrans, PlasmaUnit,
                    kb,
                    nva0-k-kb,
                    CBLAS_SADDR(zone), &a0[k+k*lda0], lda0,
                                       &a0[k+(k+kb)*lda0], lda0);

        // gemm
        plasma_complex64_t zmone = -1.0;
        for (int i = 0; i < A.mt; i++) {

            plasma_complex64_t *ai = A(i, 0);
            int mvai = plasma_tile_mview(A, i);
            int ldai = plasma_tile_mmain(A, i);

            if (i == 0) {
                cblas_zgemm(CblasColMajor,
                            CblasNoTrans, CblasNoTrans,
                            mva0-k-kb,
                            nva0-k-kb,
                            kb,
                            CBLAS_SADDR(zmone), &a0[k+kb+k*lda0], lda0,
                                                &a0[k+(k+kb)*lda0], lda0,
                            CBLAS_SADDR(zone),  &a0[(k+kb)+(k+kb)*lda0], lda0);
            }
            else {
                cblas_zgemm(CblasColMajor,
                            CblasNoTrans, CblasNoTrans,
                            mvai,
                            nva0-k-kb,
                            kb,
                            CBLAS_SADDR(zmone), &ai[k*ldai], ldai,
                                                &a0[k+(k+kb)*lda0], lda0,
                            CBLAS_SADDR(zone),  &ai[(k+kb)*ldai], ldai);           
            }
        }
pzdesc2ge(A, pA, lda);

    }

pzge2desc(pA, lda, A);

    // left pivoting
    for (int k = ib; k < imin(m, n); k += ib) {
        for (int i = k; i < imin(m, n); i++) {

            plasma_complex64_t *ai = A(i/nb, 0);
            plasma_complex64_t *ap = A((ipiv[i]-1)/nb, 0);
            int ldai = plasma_tile_mmain(A, (i/nb));
            int ldap = plasma_tile_mmain(A, (ipiv[i]-1)/nb);

            cblas_zswap(ib,
                        &ai[i%nb+(k-ib)*ldai], ldai,
                        &ap[(ipiv[i]-1)%nb+(k-ib)*ldap], ldap);
        }

    }
pzdesc2ge(A, pA, lda);
}

/******************************************************************************/
void plasma_zgetrf__(int m, int n, plasma_complex64_t *A, int lda, int *ipiv, int nb)
{
    for (int k = 0; k < imin(m, n); k++) {

        // panel
        LAPACKE_zgetrf(LAPACK_COL_MAJOR,
                       m-k, 1, &A[k+k*lda], lda, &ipiv[k]);

        for (int i = k+1; i <= imin(m, k+1); i++)
            ipiv[i-1] += k;

        // left pivoting
        LAPACKE_zlaswp(LAPACK_COL_MAJOR,
                       k,
                       &A[0], lda,
                       k+1,
                       k+1,
                       ipiv, 1);

        if (k == imin(m, n)-1)
            break;

        int l = (k+1)&(~k);
        int kb = imin(imin(m, n)-k-1, l);

        // right pivoting
        LAPACKE_zlaswp(LAPACK_COL_MAJOR,
                       kb,
                       &A[(k+1)*lda], lda,
                       (k-l+1)+1,
                       k+1,
                       ipiv, 1);

        plasma_complex64_t zone = 1.0;
        cblas_ztrsm(CblasColMajor,
                    PlasmaLeft, PlasmaLower,
                    PlasmaNoTrans, PlasmaUnit,
                    l,
                    kb,
                    CBLAS_SADDR(zone), &A[k+1-l+(k+1-l)*lda], lda,
                                       &A[k+1-l+(k+1)*lda], lda);

        plasma_complex64_t zmone = -1.0;
        cblas_zgemm(CblasColMajor,
                    CblasNoTrans, CblasNoTrans,
                    m-k-1,
                    kb,
                    l,
                    CBLAS_SADDR(zmone), &A[k+1+(k+1-l)*lda], lda,
                                        &A[k+1-l+(k+1)*lda], lda,
                    CBLAS_SADDR(zone),  &A[k+1+(k+1)*lda], lda);
    }

    // right reminder for n > m
    int k = imin(m, n);
    LAPACKE_zlaswp(LAPACK_COL_MAJOR,
                   n-k,
                   &A[k*lda], lda,
                   1,
                   k,
                   ipiv, 1);

    plasma_complex64_t zone = 1.0;
    cblas_ztrsm(CblasColMajor,
                PlasmaLeft, PlasmaLower,
                PlasmaNoTrans, PlasmaUnit,
                k,
                n-k,
                CBLAS_SADDR(zone), &A[0], lda,
                                   &A[k*lda], lda);
}

/******************************************************************************/
static void print_matrix(plasma_complex64_t *A, int m, int n)
{
    for (int j = 0; j < m; j++) {
        for (int i = 0; i < n; i++) {

            double v = cabs(A[j+i*m]);
            char c;

                 if (v < 0.0000000001) c = '.';
            else if (v == 1.0) c = '#';
            else c = 'o';

            printf ("%c ", c);
        }
        printf("\n");
    }
}

#define COMPLEX

#define A(i_, j_) A[(i_) + (size_t)lda*(j_)]

/***************************************************************************//**
 *
 * @brief Tests ZPOTRF.
 *
 * @param[in]  param - array of parameters
 * @param[out] info  - string of column labels or column values; length InfoLen
 *
 * If param is NULL and info is NULL,     print usage and return.
 * If param is NULL and info is non-NULL, set info to column labels and return.
 * If param is non-NULL and info is non-NULL, set info to column values
 * and run test.
 ******************************************************************************/
void test_zgetrf(param_value_t param[], char *info)
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
            print_usage(PARAM_NB);
        }
        else {
            // Return column labels.
            snprintf(info, InfoLen,
                     "%*s %*s %*s %*s",
                     InfoSpacing, "M",
                     InfoSpacing, "N",
                     InfoSpacing, "PadA",
                     InfoSpacing, "NB");
        }
        return;
    }
    // Return column values.
    snprintf(info, InfoLen,
             "%*d %*d %*d %*d",
             InfoSpacing, param[PARAM_M].i,
             InfoSpacing, param[PARAM_N].i,
             InfoSpacing, param[PARAM_PADA].i,
             InfoSpacing, param[PARAM_NB].i);

    //================================================================
    // Set parameters.
    //================================================================
    int m = param[PARAM_M].i;
    int n = param[PARAM_N].i;

    int lda = imax(1, m+param[PARAM_PADA].i);

    int test = param[PARAM_TEST].c == 'y';
    double tol = param[PARAM_TOL].d * LAPACKE_dlamch('E');

    //================================================================
    // Set tuning parameters.
    //================================================================
    plasma_set(PlasmaNb, param[PARAM_NB].i);

    //================================================================
    // Allocate and initialize arrays.
    //================================================================
    plasma_complex64_t *A =
        (plasma_complex64_t*)malloc((size_t)lda*n*sizeof(plasma_complex64_t));
    assert(A != NULL);

    int *IPIV = (int*)malloc((size_t)m*sizeof(int));
    assert(IPIV != NULL);

    int seed[] = {0, 0, 0, 1};
    lapack_int retval;
    retval = LAPACKE_zlarnv(1, seed, (size_t)lda*n, A);
    assert(retval == 0);

    plasma_complex64_t *Aref = NULL;
    if (test) {
        Aref = (plasma_complex64_t*)malloc(
            (size_t)lda*n*sizeof(plasma_complex64_t));
        assert(Aref != NULL);

        memcpy(Aref, A, (size_t)lda*n*sizeof(plasma_complex64_t));
    }

    //================================================================
    // Run and time PLASMA.
    //================================================================
    plasma_time_t start = omp_get_wtime();
    plasma_zgetrf_(m, n, A, lda, IPIV, param[PARAM_NB].i, param[PARAM_IB].i);
//  plasma_zgetrf(m, n, A, lda, IPIV);
    plasma_time_t stop = omp_get_wtime();
    plasma_time_t time = stop-start;

    param[PARAM_TIME].d = time;
    param[PARAM_GFLOPS].d = flops_zpotrf(n) / time / 1e9;

    //================================================================
    // Test results by comparing to a reference implementation.
    //================================================================
    if (test) {
        LAPACKE_zgetrf(
            LAPACK_COL_MAJOR,
            m, n,
            Aref, lda, IPIV);

        plasma_complex64_t zmone = -1.0;
        cblas_zaxpy((size_t)lda*n, CBLAS_SADDR(zmone), Aref, 1, A, 1);

print_matrix(A, m, n);

        double work[1];
        double Anorm = LAPACKE_zlange_work(
            LAPACK_COL_MAJOR, 'F', m, n, Aref, lda, work);

        double error = LAPACKE_zlange_work(
            LAPACK_COL_MAJOR, 'F', m, n, A, lda, work);

        if (Anorm != 0)
            error /= Anorm;

        param[PARAM_ERROR].d = error;
        param[PARAM_SUCCESS].i = error < tol;
    }

    //================================================================
    // Free arrays.
    //================================================================
    free(A);
    free(IPIV);
    if (test)
        free(Aref);
}
