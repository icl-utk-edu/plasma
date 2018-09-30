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
#include <plasma_core_blas.h>

#define A(m, n) (plasma_complex64_t*)plasma_tile_addr(A, m, n)
#define T(m, n) (plasma_complex64_t*)plasma_tile_addr(T, m, n)
#define Q(m, n) (plasma_complex64_t*)plasma_tile_addr(Q, m, n)

/***************************************************************************//**
 *  Parallel construction of Q using tile V (application to identity)
 **/
void plasma_pzunglq(plasma_desc_t A, plasma_desc_t T, plasma_desc_t Q,
                    plasma_workspace_t work,
                    plasma_sequence_t *sequence, plasma_request_t *request)
{
    // Return if failed sequence.
    if (sequence->status != PlasmaSuccess)
        return;

    // Set inner blocking from the T tile row-dimension.
    int ib = T.mb;

    for (int k = imin(A.mt, A.nt)-1; k >= 0; k--) {
        int mvak = plasma_tile_mview(A, k);
        int nvak = plasma_tile_nview(A, k);
        int nvqk = plasma_tile_nview(Q, k);
        int ldak = plasma_tile_mmain(A, k);
        for (int n = Q.nt-1; n > k; n--) {
            int nvqn = plasma_tile_nview(Q, n);
            for (int m = k; m < Q.mt; m++) {
                int mvqm = plasma_tile_mview(Q, m);
                int ldqm = plasma_tile_mmain(Q, m);
                plasma_core_omp_ztsmlq(
                    PlasmaRight, PlasmaNoTrans,
                    mvqm, Q.nb, mvqm, nvqn, mvak, ib,
                    Q(m, k), ldqm,
                    Q(m, n), ldqm,
                    A(k, n), ldak,
                    T(k, n), T.mb,
                    work,
                    sequence, request);
            }
        }
        for (int m = k; m < Q.mt; m++) {
            int mvqm = plasma_tile_mview(Q, m);
            int ldqm = plasma_tile_mmain(Q, m);
            plasma_core_omp_zunmlq(
                PlasmaRight, PlasmaNoTrans,
                mvqm, nvqk, imin(nvak, mvak), ib,
                A(k, k), ldak,
                T(k, k), T.mb,
                Q(m, k), ldqm,
                work,
                sequence, request);
        }
    }
}
