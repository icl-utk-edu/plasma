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

#include "plasma_async.h"
#include "plasma_context.h"
#include "plasma_descriptor.h"
#include "plasma_internal.h"
#include "plasma_types.h"
#include "plasma_workspace.h"
#include "core_blas.h"

#include "plasma_z.h"
#include "mkl_lapacke.h"
#include "mkl_cblas.h"

#define A(m, n) (plasma_complex64_t*)plasma_tile_addr(A, m, n)

/******************************************************************************/
static void print_matrix(plasma_complex64_t *A, int m, int n, int nb)
{
    for (int j = 0; j < m; j++) {
        for (int i = 0; i < n; i++) {

            double v = cabs(A[j+i*m]);
            char c;

                 if (v < 0.0000000001) c = '.';
            else if (v == 1.0) c = '#';
            else c = 'o';

            printf ("%c ", c);
            if ((i+1) % nb == 0)
                printf ("| ");

            if (i%nb == 1)
                i += (nb-4);
        }
        printf("\n");
        if ((j+1) % nb == 0) {
//          for (int i = 0; i < n + n/nb; i++)
            for (int i = 0; i < n/nb*5; i++)
                printf ("--");
            printf("\n");
        }
        if (j%nb == 1)
            j += (nb-4);
    }
}

/******************************************************************************/
void core_zlaswp(plasma_desc_t A, int n, int k1, int k2, int *ipiv)
{
    for (int m = k1-1; m <= k2-1; m++) {
        if (ipiv[m]-1 != m) {

            int m1 = m;
            int m2 = ipiv[m]-1;

            int nvan = plasma_tile_nview(A, n);
            int lda1 = plasma_tile_mmain(A, m1/A.mb);
            int lda2 = plasma_tile_mmain(A, m2/A.mb);

            cblas_zswap(nvan,
                        A(m1/A.mb, n) + m1%A.mb, lda1,
                        A(m2/A.mb, n) + m2%A.mb, lda2);
        }
    }
}

/******************************************************************************/
void plasma_pzgetrf(plasma_desc_t A, int *ipiv,
                    plasma_sequence_t *sequence, plasma_request_t *request)
{
    // Check sequence status.
    if (sequence->status != PlasmaSuccess) {
        plasma_request_fail(sequence, request, PlasmaErrorSequence);
        return;
    }



    plasma_complex64_t *pA = (plasma_complex64_t*)malloc(
        (size_t)A.m*A.n*sizeof(plasma_complex64_t));

    plasma_complex64_t *pB = (plasma_complex64_t*)malloc(
        (size_t)A.m*A.n*sizeof(plasma_complex64_t));



    #pragma omp parallel
    #pragma omp master
        plasma_omp_zdesc2ge(A, pA, A.m, sequence, request);



    memcpy(pB, pA, (size_t)A.m*A.n*sizeof(plasma_complex64_t));
    LAPACKE_zgetrf(LAPACK_COL_MAJOR, A.m, A.n, pB, A.m, ipiv);



    for (int k = 0; k < imin(A.mt, A.nt); k++)
    {



        #pragma omp parallel
        #pragma omp master
            plasma_omp_zdesc2ge(A, pA, A.m, sequence, request);

        // Factor the panel.
        int nvak = plasma_tile_nview(A, k);
        LAPACKE_zgetrf(LAPACK_COL_MAJOR,
                       A.m-k*A.nb, nvak,
                       &pA[k*A.nb*A.m + k*A.nb], A.m, &ipiv[k*A.nb]);

        // Adjust pivot indices.
        for (int i = k*A.nb+1; i <= A.m; i++)
            ipiv[i-1] += k*A.nb;

        #pragma omp parallel
        #pragma omp master
            plasma_omp_zge2desc(pA, A.m, A, sequence, request);



        // Pivot to the right.
        for (int n = k+1; n < A.nt; n++) {
            int nvan = plasma_tile_nview(A, n);
            int ione = 1;
            int k1 = k*A.nb+1;
            int k2 = imin(k*A.nb+A.nb, A.m);
            core_zlaswp(A, n, k1, k2, ipiv);
        }

        #pragma omp parallel
        #pragma omp master
        {
            for (int n = k+1; n < A.nt; n++) {
                int mvak = plasma_tile_mview(A, k);
                int nvan = plasma_tile_nview(A, n);
                int ldak = plasma_tile_mmain(A, k);
                core_omp_ztrsm(
                    PlasmaLeft, PlasmaLower,
                    PlasmaNoTrans, PlasmaUnit,
                    mvak, nvan,
                    1.0, A(k, k), ldak,
                         A(k, n), ldak,
                    sequence, request);
            }
        }

        #pragma omp parallel
        #pragma omp master
        {
            for (int n = k+1; n < A.nt; n++) {
                for (int m = k+1; m < A.mt; m++) {
                    int mvam = plasma_tile_mview(A, m);
                    int nvan = plasma_tile_nview(A, n);
                    int ldam = plasma_tile_mmain(A, m);
                    int ldak = plasma_tile_mmain(A, k);
                    core_omp_zgemm(
                        PlasmaNoTrans, PlasmaNoTrans,
                        mvam, nvan, A.nb,
                        -1.0, A(m, k), ldam,
                              A(k, n), ldak,
                        1.0,  A(m, n), ldam,
                        sequence, request);
                }
            }
        }

    }

    // Pivot to the left.
    for (int k = 1; k < imin(A.mt, A.nt); k++) {
        int k1 = k*A.nb+1;
        int k2 = imin(A.m, A.n);
        int ione = 1;
        core_zlaswp(A, k-1, k1, k2, ipiv);
    }



    #pragma omp parallel
    #pragma omp master
        plasma_omp_zdesc2ge(A, pA, A.m, sequence, request);

    plasma_complex64_t zmone = -1.0;
    cblas_zaxpy((size_t)A.m*A.n, CBLAS_SADDR(zmone), pA, 1, pB, 1);
    print_matrix(pB, A.m, A.n, A.nb);

    #pragma omp parallel
    #pragma omp master
        plasma_omp_zge2desc(pA, A.m, A, sequence, request);

    free(pA);
    free(pB);
}

/*
        // Pivot to the left as you go.
        int n = k*A.nb;
        int k1 = k*A.nb+1;
        int k2 = (k+1)*A.nb;
        int ione = 1;
        zlaswp(&n, pA, &A.m, &k1, &k2, ipiv, &ione);
*/
/*
        // Apply triangular solve to the block of U.
        for (int n = k+1; n < A.nt; n++) {
            plasma_complex64_t zone = 1.0;
            cblas_ztrsm(
                CblasColMajor,
                CblasLeft, CblasLower, CblasNoTrans, CblasUnit,
                A.nb, A.nb,
                CBLAS_SADDR(zone), &pA[k*A.nb + k*A.nb*A.m], A.m,
                                   &pA[k*A.nb + n*A.nb*A.m], A.m);
        }
*/
/*
        plasma_complex64_t mzone = -1.0;
        plasma_complex64_t zone  =  1.0;
        for (int n = k+1; n < A.nt; n++)
            for (int m = k+1; m < A.mt; m++) {
                cblas_zgemm(
                    CblasColMajor,
                    CblasNoTrans, CblasNoTrans,
                    A.nb, A.nb, A.nb,
                    CBLAS_SADDR(mzone), &pA[m*A.nb + k*A.nb*A.m], A.m,
                                        &pA[k*A.nb + n*A.nb*A.m], A.m,
                    CBLAS_SADDR(zone),  &pA[m*A.nb + n*A.nb*A.m], A.m);
            }
*/
