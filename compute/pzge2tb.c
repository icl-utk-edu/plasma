/**
 *
 * @file
 *
 *  PLASMA is a software package provided by:
 *  University of Tennessee, US,
 *  University of Manchester, UK.
 *
 * @precisions normal z -> s d  c
 *
 **/

#include "plasma_async.h"
#include "plasma_context.h"
#include "plasma_descriptor.h"
#include "plasma_internal.h"
#include "plasma_types.h"
#include "plasma_workspace.h"
#include "plasma_core_blas.h"

#define A(m, n) ((plasma_complex64_t*) plasma_tile_addr(A, m, n))
#define T(m, n) ((plasma_complex64_t*) plasma_tile_addr(T, m, n))
/***************************************************************************//**
 *  Parallel tile BAND Bidiagonal Reduction - panel-based version
 **/
void plasma_pzge2tb(
    plasma_desc_t A, plasma_desc_t T,
                    plasma_workspace_t work,
                    plasma_sequence_t *sequence, plasma_request_t *request)
{
    // Return if failed sequence.
    if (sequence->status != PlasmaSuccess)
        return;

    if (A.m >= A.n) {
        for (int k = 0; k < A.nt; k++) {
            int mvak = plasma_tile_mview(A, k);
            int nvak = plasma_tile_nview(A, k);

            // QR factorization of the k-th tile-column
            plasma_pzgeqrf(
                plasma_desc_view(A, k*A.mb, k*A.nb, A.m-k*A.mb, nvak),
                plasma_desc_view(T, k*T.mb, k*T.nb, T.m-k*T.mb, nvak),
                work,
                sequence, request);

            if (k+1 < A.nt) {
                // do not apply update in the last tile-column
                plasma_pzunmqr(
                    PlasmaLeft,
                    Plasma_ConjTrans,
                    plasma_desc_view(A, k*A.mb,     k*A.nb, A.m-k*A.mb, nvak),
                    plasma_desc_view(T, k*T.mb,     k*T.nb, T.m-k*T.mb, nvak),
                    plasma_desc_view(A, k*A.mb, (k+1)*A.nb, A.m-k*A.mb, A.n-(k+1)*A.nb),
                    work,
                    sequence, request);
            }

            if (k+1 < A.nt) {
                // LQ factorization of the k-th tile-row, shifted by 1 tile to the right
                plasma_pzgelqf(
                    plasma_desc_view(A, k*A.mb, (k+1)*A.nb, mvak, A.n-(k+1)*A.nb),
                    plasma_desc_view(T, k*T.mb, (k+1)*T.nb, T.mb, T.n-(k+1)*T.nb),
                    work,
                    sequence, request);

                // update of the (k+1:mt-1) tile-rows
                plasma_pzunmlq(
                    PlasmaRight, Plasma_ConjTrans,
                    plasma_desc_view(A,     k*A.mb, (k+1)*A.nb, mvak,           A.n-(k+1)*A.nb),
                    plasma_desc_view(T,     k*T.mb, (k+1)*T.nb, T.mb,           T.n-(k+1)*T.nb),
                    plasma_desc_view(A, (k+1)*A.mb, (k+1)*A.nb, A.m-(k+1)*A.mb, A.n-(k+1)*A.nb),
                    work,
                    sequence, request);
            }
        }
    }
    else { // A.m < A.n (more tile-columns than tile-rows)
        for (int k = 0; k < A.mt; k++) {
            int mvak = plasma_tile_mview(A, k);
            int nvak = plasma_tile_nview(A, k);

            // LQ factorization of the k-th tile-row
            plasma_pzgelqf(
                plasma_desc_view(A, k*A.mb, k*A.nb, mvak, A.n-k*A.nb),
                plasma_desc_view(T, k*T.mb, k*T.nb, T.mb, T.n-k*T.nb),
                work,
                sequence, request);

            // update of the (k+1:mt-1) tile-rows
            if (k+1 < A.mt) {
                // do not apply update in the last tile-row
                plasma_pzunmlq(
                    PlasmaRight, Plasma_ConjTrans,
                    plasma_desc_view(A,     k*A.mb, k*A.nb, mvak,           A.n-k*A.nb),
                    plasma_desc_view(T,     k*T.mb, k*T.nb, T.mb,           T.n-k*T.nb),
                    plasma_desc_view(A, (k+1)*A.mb, k*A.nb, A.m-(k+1)*A.mb, A.n-k*A.nb),
                    work,
                    sequence, request);
            }

            if (k+1 < A.mt) {
                // QR factorization of the k-th tile-column, shifted by 1 tile to the bottom
                plasma_pzgeqrf(
                     plasma_desc_view(A, (k+1)*A.mb, k*A.nb, A.m-(k+1)*A.mb, nvak),
                     plasma_desc_view(T, (k+1)*T.mb, k*T.nb, T.m-(k+1)*T.mb, nvak),
                     work,
                     sequence, request);

                plasma_pzunmqr(
                    PlasmaLeft, Plasma_ConjTrans,
                    plasma_desc_view(A, (k+1)*A.mb,     k*A.nb, A.m-(k+1)*A.mb, nvak),
                    plasma_desc_view(T, (k+1)*T.mb,     k*T.nb, T.m-(k+1)*T.mb, nvak),
                    plasma_desc_view(A, (k+1)*A.mb, (k+1)*A.nb, A.m-(k+1)*A.mb, A.n-(k+1)*A.nb),
                    work,
                    sequence, request);
            }
        }
    }
}
