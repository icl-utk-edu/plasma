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
#define Q(m, n) (plasma_complex64_t*)plasma_tile_addr(Q, m, n)
#define T(m, n) (plasma_complex64_t*)plasma_tile_addr(T, m, n)

/***************************************************************************//**
 *  Parallel construction of Q using tile V (application to identity)
 **/
void plasma_pzunglq(plasma_desc_t A, plasma_desc_t Q, plasma_desc_t T,
                    plasma_workspace_t *work,
                    plasma_sequence_t *sequence, plasma_request_t *request)
{
    int k, m, n;
    int ldak, ldqm;
    int tempnn, tempmm, tempkmin, tempkn;
    int tempAkm, tempAkn;
    int minmnt;

    // Check sequence status.
    if (sequence->status != PlasmaSuccess) {
        plasma_request_fail(sequence, request, PlasmaErrorSequence);
        return;
    }

    // Set inner blocking from the T tile row-dimension.
    int ib = T.mb;

    minmnt = imin(A.mt, A.nt);
    for (k = minmnt-1; k >= 0; k--) {
        tempAkm  = plasma_tile_mdim(A, k);
        tempAkn  = plasma_tile_ndim(A, k);
        tempkmin = imin(tempAkn, tempAkm);
        tempkn   = plasma_tile_ndim(Q, k);
        ldak     = plasma_tile_mdim(A, k);
        for (n = Q.nt-1; n > k; n--) {
            tempnn = plasma_tile_ndim(Q, n);
            for (m = k; m < Q.mt; m++) {
                tempmm = plasma_tile_mdim(Q, m);
                ldqm   = plasma_tile_mdim(Q, m);
                core_omp_ztsmlq(
                    PlasmaRight, PlasmaNoTrans,
                    tempmm, Q.nb, tempmm, tempnn, tempAkm, ib, T.nb,
                    Q(m, k), ldqm,
                    Q(m, n), ldqm,
                    A(k, n), ldak,
                    T(k, n), T.mb,
                    work,
                    sequence, request);
            }
        }
        for (m = k; m < Q.mt; m++) {
            tempmm = plasma_tile_mdim(Q, m);
            ldqm   = plasma_tile_mdim(Q, m);
            core_omp_zunmlq(
                PlasmaRight, PlasmaNoTrans,
                tempmm, tempkn, tempkmin, ib, T.nb,
                A(k, k), ldak,
                T(k, k), T.mb,
                Q(m, k), ldqm,
                work,
                sequence, request);
        }
    }
}
