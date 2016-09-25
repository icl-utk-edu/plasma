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
void plasma_pzungqr(plasma_desc_t A, plasma_desc_t Q, plasma_desc_t T,
                    plasma_workspace_t *work,
                    plasma_sequence_t *sequence, plasma_request_t *request)
{
    int k, m, n;
    int ldak, ldqk, ldam, ldqm;
    int tempmm, tempnn, tempkmin, tempkm;
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
        tempkm   = plasma_tile_mdim(Q, k);
        ldak     = plasma_tile_mdim(A, k);
        ldqk     = plasma_tile_mdim(Q, k);
        for (m = Q.mt - 1; m > k; m--) {
            tempmm = plasma_tile_mdim(Q, m);
            ldam   = plasma_tile_mdim(A, m);
            ldqm   = plasma_tile_mdim(Q, m);
            for (n = k; n < Q.nt; n++) {
                tempnn = plasma_tile_ndim(Q, n);
                core_omp_ztsmqr(
                    PlasmaLeft, PlasmaNoTrans,
                    Q.mb, tempnn, tempmm, tempnn, tempAkn, ib, T.nb,
                    Q(k, n), ldqk,
                    Q(m, n), ldqm,
                    A(m, k), ldam,
                    T(m, k), T.mb,
                    work,
                    sequence, request);
            }
        }
        for (n = k; n < Q.nt; n++) {
            tempnn = plasma_tile_ndim(Q, n);
            core_omp_zunmqr(
                PlasmaLeft, PlasmaNoTrans,
                tempkm, tempnn, tempkmin, ib, T.nb,
                A(k, k), ldak,
                T(k, k), T.mb,
                Q(k, n), ldqk,
                work,
                sequence, request);
        }
    }
}
