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
#include "plasma_tree.h"
#include <plasma_core_blas.h>

#define A(m, n) (plasma_complex64_t*)plasma_tile_addr(A, m, n)
#define Q(m, n) (plasma_complex64_t*)plasma_tile_addr(Q, m, n)
#define T(m, n) (plasma_complex64_t*)plasma_tile_addr(T, m, n)
#define T2(m, n) (plasma_complex64_t*)plasma_tile_addr(T, m, n+(T.nt/2))
/***************************************************************************//**
 *  Parallel construction of Q using tile V (application to identity)
 *  based on a tree Householder reduction.
 **/
void plasma_pzungqr_tree(plasma_desc_t A, plasma_desc_t T, plasma_desc_t Q,
                         plasma_workspace_t work,
                         plasma_sequence_t *sequence, plasma_request_t *request)
{
    // Return if failed sequence.
    if (sequence->status != PlasmaSuccess)
        return;

    // Precompute order of QR operations.
    int *operations = NULL;
    int num_operations;
    plasma_tree_operations(A.mt, A.nt, &operations, &num_operations,
                           sequence, request);

    // Set inner blocking from the T tile row-dimension.
    int ib = T.mb;

    for (int iop = num_operations-1; iop >= 0; iop--) {
        int j, k, kpiv;
        plasma_enum_t kernel;
        plasma_tree_get_operation(operations, iop, &kernel,
                                  &j, &k, &kpiv);

        int nvaj = plasma_tile_nview(A, j);
        int mvak = plasma_tile_mview(A, k);
        int ldak = plasma_tile_mmain(A, k);
        int mvqk = plasma_tile_mview(Q, k);
        int ldqk = plasma_tile_mmain(Q, k);

        if (kernel == PlasmaGeKernel) {
            // triangularization
            for (int n = j; n < Q.nt; n++) {
                int nvqn = plasma_tile_nview(Q, n);

                plasma_core_omp_zunmqr(PlasmaLeft, PlasmaNoTrans,
                                mvqk, nvqn, imin(mvak, nvaj), ib,
                                A(k, j), ldak,
                                T(k, j), T.mb,
                                Q(k, n), ldqk,
                                work,
                                sequence, request);
            }
        }
        else if (kernel == PlasmaTtKernel) {
            // elimination of the tile
            int mvakpiv = plasma_tile_mview(A, kpiv);
            int mvqkpiv = plasma_tile_mview(Q, kpiv);
            int ldqkpiv = plasma_tile_mmain(Q, kpiv);

            for (int n = j; n < Q.nt; n++) {
                int nvqn = plasma_tile_nview(Q, n);

                plasma_core_omp_zttmqr(
                    PlasmaLeft, PlasmaNoTrans,
                    mvqkpiv, nvqn, mvqk, nvqn, imin(mvakpiv+mvak, nvaj), ib,
                    Q(kpiv, n), ldqkpiv,
                    Q(k,    n), ldqk,
                    A(k,    j), ldak,
                    T2(k,   j), T.mb,
                    work,
                    sequence, request);
            }
        }
        else if (kernel == PlasmaTsKernel) {
            // elimination of the tile
            int mvakpiv = plasma_tile_mview(A, kpiv);
            int mvqkpiv = plasma_tile_mview(Q, kpiv);
            int ldqkpiv = plasma_tile_mmain(Q, kpiv);

            for (int n = j; n < Q.nt; n++) {
                int nvqn = plasma_tile_nview(Q, n);

                plasma_core_omp_ztsmqr(
                    PlasmaLeft, PlasmaNoTrans,
                    mvqkpiv, nvqn, mvqk, nvqn, imin(mvakpiv+mvak, nvaj), ib,
                    Q(kpiv, n), ldqkpiv,
                    Q(k,    n), ldqk,
                    A(k,    j), ldak,
                    T2(k,   j), T.mb,
                    work,
                    sequence, request);
            }
        }
        else {
            plasma_error("illegal kernel");
            plasma_request_fail(sequence, request, PlasmaErrorIllegalValue);
        }
    }

    free(operations);
}
