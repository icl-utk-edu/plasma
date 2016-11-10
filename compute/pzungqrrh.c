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
#include "plasma_rh_tree.h"
#include "core_blas.h"

#define A(m, n) (plasma_complex64_t*)plasma_tile_addr(A, m, n)
#define Q(m, n) (plasma_complex64_t*)plasma_tile_addr(Q, m, n)
#define T(m, n) (plasma_complex64_t*)plasma_tile_addr(T, m, n)
#define T2(m, n) (plasma_complex64_t*)plasma_tile_addr(T, m, n+(T.nt/2))
/***************************************************************************//**
 *  Parallel construction of Q using tile V (application to identity)
 *  based on a tree Householder reduction.
 **/
void plasma_pzungqrrh(plasma_desc_t A, plasma_desc_t T, plasma_desc_t Q,
                      plasma_workspace_t work,
                      plasma_sequence_t *sequence, plasma_request_t *request)
{
    static const int debug = 0;

    // Check sequence status.
    if (sequence->status != PlasmaSuccess) {
        plasma_request_fail(sequence, request, PlasmaErrorSequence);
        return;
    }

    // Precompute order of QR operations.
    int *operations = NULL;
    int noperations;
    plasma_rh_tree_operations(A.mt, A.nt, &operations, &noperations);

    // Set inner blocking from the T tile row-dimension.
    int ib = T.mb;

    for (int iop = noperations-1; iop >= 0; iop--) {
        int j, k, kpiv;
        plasma_enum_t kernel;
        plasma_rh_tree_operation_get(operations, iop, &kernel,
                                     &j, &k, &kpiv);

        int nvaj = plasma_tile_nview(A, j);
        int mvak = plasma_tile_mview(A, k);
        int ldak = plasma_tile_mmain(A, k);
        int mvqk = plasma_tile_mview(Q, k);
        int ldqk = plasma_tile_mmain(Q, k);

        if (kernel == PlasmaGEKernel) {
            // triangularization
            for (int n = j; n < Q.nt; n++) {
                int nvqn = plasma_tile_nview(Q, n);

                if (debug) printf("UNMQR (%d,%d,%d) ", k, j, n);
                core_omp_zunmqr(PlasmaLeft, PlasmaNoTrans,
                                mvqk, nvqn, 
                                imin(mvak, nvaj), ib,
                                A(k, j), ldak,
                                T(k, j), T.mb,
                                Q(k, n), ldqk,
                                work,
                                sequence, request);
            }

            if (debug) printf("\n ");
        }
        else if (kernel == PlasmaTTKernel) {
            // elimination of the tile
            int mvqkpiv = plasma_tile_mview(Q, kpiv);
            int ldqkpiv = plasma_tile_mmain(Q, kpiv);

            for (int n = j; n < Q.nt; n++) {
                int nvqn = plasma_tile_nview(Q, n);

                if (debug) printf("TTMQR (%d,%d,%d,%d) ", k, kpiv, j, n);
                core_omp_zttmqr(
                    PlasmaLeft, PlasmaNoTrans,
                    mvqkpiv, nvqn, mvqk, nvqn, nvaj, ib,
                    Q(kpiv, n), ldqkpiv,
                    Q(k,    n), ldqk,
                    A(k,    j), ldak,
                    T2(k,   j), T.mb,
                    work,
                    sequence, request);
            }

            if (debug) printf("\n ");
        }
        else {
            plasma_error("illegal kernel");
            plasma_request_fail(sequence, request, PlasmaErrorIllegalValue);
        }
    }

    free(operations);
}
