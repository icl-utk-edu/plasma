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

#define A(m, n) (plasma_complex64_t*)plasma_tile_addr(A, m, n)

/******************************************************************************/
void plasma_pzgetrf(plasma_desc_t A, int *ipiv,
                    plasma_sequence_t *sequence, plasma_request_t *request)
{
#pragma omp parallel
#pragma omp master
{
    // Check sequence status.
    if (sequence->status != PlasmaSuccess) {
        plasma_request_fail(sequence, request, PlasmaErrorSequence);
//      return;
    }

    // Read parameters from the context.
    plasma_context_t *plasma = plasma_context_self();
    int ib = plasma->ib;
    int num_panel_threads = plasma->num_panel_threads;
    plasma_barrier_t *barrier = &plasma->barrier;

    for (int k = 0; k < imin(A.mt, A.nt); k++) {

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
            for (int i = 0; i < num_panel_threads; i++) {
                #pragma omp task priority(1)
                {
                    plasma_desc_t view =
                        plasma_desc_view(A, k*A.mb, k*A.nb, A.m-k*A.mb, nvak);

                    core_zgetrf(view, &ipiv[k*A.mb], ib, i,
                                num_panel_threads, barrier);
                }
            }
            #pragma omp taskwait

            for (int i = k*A.mb+1; i <= A.m; i++)
                ipiv[i-1] += k*A.mb;
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
                             depend(inout:a21[0:lda21*nvan]) \
                             priority(n == k+1)
            {
                // laswp
                int k1 = k*A.mb+1;
                int k2 = imin(k*A.mb+A.mb, A.m);
                plasma_desc_t view = plasma_desc_view(A, 0, n*A.nb, A.m, nvan);
                core_zlaswp(view, k1, k2, ipiv);

                // trsm
                core_ztrsm(PlasmaLeft, PlasmaLower,
                           PlasmaNoTrans, PlasmaUnit,
                           mvak, nvan,
                           1.0, A(k, k), ldak,
                                A(k, n), ldak);

                // gemm
                for (int m = k+1; m < A.mt; m++) {

                    int mvam = plasma_tile_mview(A, m);
                    int ldam = plasma_tile_mmain(A, m);

                    #pragma omp task priority(n == k+1)
                    {
                        core_zgemm(
                            PlasmaNoTrans, PlasmaNoTrans,
                            mvam, nvan, A.nb,
                            -1.0, A(m, k), ldam,
                                  A(k, n), ldak,
                            1.0,  A(m, n), ldam);
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
            plasma_desc_t view = plasma_desc_view(A, 0, (k-1)*A.nb, A.m, A.nb);
            core_zlaswp(view, k1, k2, ipiv);
        }
    }
}
}
