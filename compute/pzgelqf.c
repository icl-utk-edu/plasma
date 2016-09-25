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
    int k, m, n;
    int ldak, ldam;
    int tempkm, tempkn, tempmm, tempnn;

    // Check sequence status.
    if (sequence->status != PlasmaSuccess) {
        plasma_request_fail(sequence, request, PlasmaErrorSequence);
        return;
    }

    // Set inner blocking from the T tile row-dimension.
    int ib = T.mb;

    for (k = 0; k < imin(A.mt, A.nt); k++) {
        tempkm = plasma_tile_mdim(A, k);
        tempkn = plasma_tile_ndim(A, k);
        ldak   = plasma_tile_mdim(A, k);
        core_omp_zgelqt(
            tempkm, tempkn, ib, T.nb,
            A(k, k), ldak,
            T(k, k), T.mb,
            work,
            sequence, request);

        for (m = k+1; m < A.mt; m++) {
            tempmm = plasma_tile_mdim(A, m);
            ldam   = plasma_tile_mdim(A, m);
            // Plasma_ConjTrans will be converted to PlasmaTrans in
            // automatic datatype conversion, which is what we
            // want here.
            // PlasmaConjTrans is protected from this conversion.
            core_omp_zunmlq(
                PlasmaRight, Plasma_ConjTrans,
                tempmm, tempkn, tempkn, ib, T.nb,
                A(k, k), ldak,
                T(k, k), T.mb,
                A(m, k), ldam,
                work,
                sequence, request);
        }
        for (n = k+1; n < A.nt; n++) {
            tempnn = plasma_tile_ndim(A, n);
            core_omp_ztslqt(
                tempkm, tempnn, ib, T.nb,
                A(k, k), ldak,
                A(k, n), ldak,
                T(k, n), T.mb,
                work,
                sequence, request);

            for (m = k+1; m < A.mt; m++) {
                tempmm = plasma_tile_mdim(A, m);
                ldam   = plasma_tile_mdim(A, m);
                core_omp_ztsmlq(
                    PlasmaRight, Plasma_ConjTrans,
                    tempmm, A.nb, tempmm, tempnn, A.mb, ib, T.nb,
                    A(m, k), ldam,
                    A(m, n), ldam,
                    A(k, n), ldak,
                    T(k, n), T.mb,
                    work,
                    sequence, request);
            }
        }
    }
}
