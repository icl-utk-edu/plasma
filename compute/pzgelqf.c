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
#define T(m, n) (plasma_complex64_t*)plasma_tile_addr(T, m, n)
 
/***************************************************************************//**
 *  Parallel tile LQ factorization - dynamic scheduling
 * @see plasma_omp_zgelqf
 **/
void plasma_pzgelqf(plasma_desc_t A, plasma_desc_t T,
                    plasma_workspace_t *work,
                    plasma_sequence_t *sequence, plasma_request_t *request)
{
    // Check sequence status.
    if (sequence->status != PlasmaSuccess) {
        plasma_request_fail(sequence, request, PlasmaErrorSequence);
        return;
    }

    // Set inner blocking from the T tile row-dimension.
    int ib = T.mb;

    for (int k = 0; k < imin(A.mt, A.nt); k++) {
        int mdimak = plasma_tile_mdim(A, k);
        int ndimak = plasma_tile_ndim(A, k);
        core_omp_zgelqt(
            mdimak, ndimak, ib, T.nb,
            A(k, k), mdimak,
            T(k, k), T.mb,
            work,
            sequence, request);

        for (int m = k+1; m < A.mt; m++) {
            int mdimam = plasma_tile_mdim(A, m);
            core_omp_zunmlq(
                PlasmaRight, Plasma_ConjTrans,
                mdimam, ndimak, ndimak, ib, T.nb,
                A(k, k), mdimak,
                T(k, k), T.mb,
                A(m, k), mdimam,
                work,
                sequence, request);
        }
        for (int n = k+1; n < A.nt; n++) {
            int ndiman = plasma_tile_ndim(A, n);
            core_omp_ztslqt(
                mdimak, ndiman, ib, T.nb,
                A(k, k), mdimak,
                A(k, n), mdimak,
                T(k, n), T.mb,
                work,
                sequence, request);

            for (int m = k+1; m < A.mt; m++) {
                int mdimam = plasma_tile_mdim(A, m);
                core_omp_ztsmlq(
                    PlasmaRight, Plasma_ConjTrans,
                    mdimam, A.nb, mdimam, ndiman, A.mb, ib, T.nb,
                    A(m, k), mdimam,
                    A(m, n), mdimam,
                    A(k, n), mdimak,
                    T(k, n), T.mb,
                    work,
                    sequence, request);
            }
        }
    }
}
