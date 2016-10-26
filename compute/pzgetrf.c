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

#include "../trace/trace.h"

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
void pzdesc2ge(plasma_desc_t A, plasma_complex64_t *pA, int lda)
{
    plasma_complex64_t *f77;
    plasma_complex64_t *bdl;

    int x1, y1;
    int x2, y2;
    int n, m, ldt;

    for (m = 0; m < A.mt; m++) {
        ldt = plasma_tile_mmain(A, m);
        for (n = 0; n < A.nt; n++) {
            x1 = n == 0 ? A.j%A.nb : 0;
            y1 = m == 0 ? A.i%A.mb : 0;
            x2 = n == A.nt-1 ? (A.j+A.n-1)%A.nb+1 : A.nb;
            y2 = m == A.mt-1 ? (A.i+A.m-1)%A.mb+1 : A.mb;

            f77 = &pA[(size_t)A.nb*lda*n + (size_t)A.mb*m];
            bdl = (plasma_complex64_t*)plasma_tile_addr(A, m, n);

            core_zlacpy(PlasmaGeneral,
                        y2-y1, x2-x1,
                        &(bdl[x1*A.nb+y1]), ldt,
                        &(f77[x1*lda+y1]), lda);
        }
    }
}

/******************************************************************************/
void pzge2desc(plasma_complex64_t *pA, int lda, plasma_desc_t A)
{
    plasma_complex64_t *f77;
    plasma_complex64_t *bdl;

    int x1, y1;
    int x2, y2;
    int n, m, ldt;

    for (m = 0; m < A.mt; m++) {
        ldt = plasma_tile_mmain(A, m);
        for (n = 0; n < A.nt; n++) {
            x1 = n == 0 ? A.j%A.nb : 0;
            y1 = m == 0 ? A.i%A.mb : 0;
            x2 = n == A.nt-1 ? (A.j+A.n-1)%A.nb+1 : A.nb;
            y2 = m == A.mt-1 ? (A.i+A.m-1)%A.mb+1 : A.mb;

            f77 = &pA[(size_t)A.nb*lda*n + (size_t)A.mb*m];
            bdl = (plasma_complex64_t*)plasma_tile_addr(A, m, n);

            core_zlacpy(PlasmaGeneral,
                        y2-y1, x2-x1,
                        &(f77[x1*lda+y1]), lda,
                        &(bdl[x1*A.nb+y1]), ldt);
        }
    }
}

/******************************************************************************/
void core_zlaswp(plasma_desc_t A, int k1, int k2, int *ipiv)
{
    for (int m = k1-1; m <= k2-1; m++) {
        if (ipiv[m]-1 != m) {

            int m1 = m;
            int m2 = ipiv[m]-1;

            int lda1 = plasma_tile_mmain(A, m1/A.mb);
            int lda2 = plasma_tile_mmain(A, m2/A.mb);

            cblas_zswap(A.n,
                        A(m1/A.mb, 0) + m1%A.mb, lda1,
                        A(m2/A.mb, 0) + m2%A.mb, lda2);
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

trace_init();



#pragma omp parallel
#pragma omp master
{
    for (int k = 0; k < imin(A.mt, A.nt); k++)
    {
        plasma_complex64_t *a00 = A(k, k);
        plasma_complex64_t *a20 = A(A.mt-1, k);

        int ma00k = (A.mt-k-1)*A.mb;
        int na00k = plasma_tile_nmain(A, k);
        int lda20 = plasma_tile_mmain(A, A.mt-1);

        int nvak = plasma_tile_nview(A, k);
        int mvak = plasma_tile_mview(A, k);
        int ldak = plasma_tile_mmain(A, k);

        // panel
        #pragma omp task depend(inout:a00[0:ma00k*na00k]) \
                         depend(inout:a20[0:lda20*nvak]) \
                         depend(out:ipiv[k*A.mb:mvak]) \
                         priority(1)
        {
            trace_event_start();

            pzdesc2ge(plasma_desc_view(A, k*A.mb, k*A.nb, A.m-k*A.mb, nvak),
                      &pA[k*A.mb*A.m + k*A.mb], A.m);

            LAPACKE_zgetrf(LAPACK_COL_MAJOR,
                           A.m-k*A.mb, nvak,
                           &pA[k*A.mb*A.m + k*A.mb], A.m, &ipiv[k*A.mb]);

            for (int i = k*A.mb+1; i <= A.m; i++)
                ipiv[i-1] += k*A.mb;

            pzge2desc(&pA[k*A.mb*A.m + k*A.mb], A.m,
                      plasma_desc_view(A, k*A.mb, k*A.nb, A.m-k*A.mb, nvak));

            trace_event_stop(Tan);
        }

        // update
        for (int n = k+1; n < A.nt; n++) {

            plasma_complex64_t *a01 = A(k, n);
            plasma_complex64_t *a11 = A(k+1, n);
            plasma_complex64_t *a21 = A(A.mt-1, n);

            int ma11k = (A.mt-k-2)*A.mb;
            int na11n = plasma_tile_nmain(A, n);
            int lda21 = plasma_tile_mmain(A, A.mt-1);

            int nvan = plasma_tile_nview(A, n);

            #pragma omp task depend(in:a00[0:ma00k*na00k]) \
                             depend(in:a20[0:lda20*nvak]) \
                             depend(in:ipiv[k*A.mb:mvak]) \
                             depend(inout:a01[0:ldak*nvan]) \
                             depend(inout:a11[0:ma11k*na11n]) \
                             depend(inout:a21[0:lda21*nvan])
            {
                // laswp
                int k1 = k*A.mb+1;
                int k2 = imin(k*A.mb+A.mb, A.m);
                trace_event_start();
                core_zlaswp(plasma_desc_view(A, 0, n*A.nb, A.m, nvan), k1, k2, ipiv);
                trace_event_stop(RoyalBlue);

                // trsm
                trace_event_start();
                core_ztrsm(PlasmaLeft, PlasmaLower,
                           PlasmaNoTrans, PlasmaUnit,
                           mvak, nvan,
                           1.0, A(k, k), ldak,
                                A(k, n), ldak);
                trace_event_stop(MediumPurple);

                // gemm
                for (int m = k+1; m < A.mt; m++) {
                    int mvam = plasma_tile_mview(A, m);
                    int ldam = plasma_tile_mmain(A, m);

                    #pragma omp task priority(n == k+1)
                    {
                        trace_event_start();
                        core_zgemm(
                            PlasmaNoTrans, PlasmaNoTrans,
                            mvam, nvan, A.nb,
                            -1.0, A(m, k), ldam,
                                  A(k, n), ldak,
                            1.0,  A(m, n), ldam);
                        trace_event_stop(n == k+1 ? Lime : MediumAquamarine);
                    }
                }
                #pragma omp taskwait
            }
        }
    }

    // pivoting to the left
    for (int k = 1; k < imin(A.mt, A.nt); k++) {

        plasma_complex64_t *akk = A(k, k);
        int makk = (A.mt-k-1)*A.mb;
        int nakk = plasma_tile_nmain(A, k);

        #pragma omp task depend(in:ipiv[(imin(A.mt, A.nt)-1)*A.mb]) \
                         depend(inout:akk[0:makk*nakk])
        {
            int k1 = k*A.mb+1;
            int k2 = imin(A.m, A.n);
            int ione = 1;
            trace_event_start();
            core_zlaswp(plasma_desc_view(A, 0, (k-1)*A.nb, A.m, A.nb), k1, k2, ipiv);
            trace_event_stop(DodgerBlue);
        }
    }
}



trace_write("../trace.svg");

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
