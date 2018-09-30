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
void plasma_pzunglq_tree(plasma_desc_t A, plasma_desc_t T, plasma_desc_t Q,
                         plasma_workspace_t work,
                         plasma_sequence_t *sequence, plasma_request_t *request)
{
    // Return if failed sequence.
    if (sequence->status != PlasmaSuccess)
        return;

    // Precompute order of LQ operations - compute it as for QR
    // and transpose it.
    int *operations = NULL;
    int num_operations;
    // Transpose m and n to reuse the QR tree.
    plasma_tree_operations(A.nt, A.mt, &operations, &num_operations,
                           sequence, request);

    // Set inner blocking from the T tile row-dimension.
    int ib = T.mb;

    for (int iop = num_operations-1; iop >= 0; iop--) {
        int j, k, kpiv;
        plasma_enum_t kernel;
        plasma_tree_get_operation(operations, iop,
                                  &kernel, &j, &k, &kpiv);

        int mvaj = plasma_tile_mview(A, j);
        int nvak = plasma_tile_nview(A, k);
        int ldaj = plasma_tile_mmain(A, j);

        int nvqk = plasma_tile_nview(Q, k);

        if (kernel == PlasmaGeKernel) {
            // triangularization
            for (int m = j; m < Q.mt; m++) {
                int mvqm = plasma_tile_mview(Q, m);
                int ldqm = plasma_tile_mmain(Q, m);

                plasma_core_omp_zunmlq(PlasmaRight, PlasmaNoTrans,
                                mvqm, nvqk,
                                imin(mvaj, nvak), ib,
                                A(j, k), ldaj,
                                T(j, k), T.mb,
                                Q(m, k), ldqm,
                                work,
                                sequence, request);
            }
        }
        else if (kernel == PlasmaTtKernel) {
            // elimination of the tile
            int nvqkpiv = plasma_tile_nview(Q, kpiv);
            int nvakpiv = plasma_tile_nview(A, kpiv);

            for (int m = j; m < Q.mt; m++) {
                int mvqm = plasma_tile_mview(Q, m);
                int ldqm = plasma_tile_mmain(Q, m);

                plasma_core_omp_zttmlq(
                    PlasmaRight, PlasmaNoTrans,
                    mvqm, nvqkpiv, mvqm, nvqk, imin(mvaj, nvakpiv+nvak), ib,
                    Q(m, kpiv), ldqm,
                    Q(m, k),    ldqm,
                    A(j, k),    ldaj,
                    T2(j, k),   T.mb,
                    work,
                    sequence, request);
            }
        }
        else if (kernel == PlasmaTsKernel) {
            // elimination of the tile
            int nvqkpiv = plasma_tile_nview(Q, kpiv);
            int nvakpiv = plasma_tile_nview(A, kpiv);

            for (int m = j; m < Q.mt; m++) {
                int mvqm = plasma_tile_mview(Q, m);
                int ldqm = plasma_tile_mmain(Q, m);

                plasma_core_omp_ztsmlq(
                    PlasmaRight, PlasmaNoTrans,
                    mvqm, nvqkpiv, mvqm, nvqk, imin(mvaj, nvakpiv+nvak), ib,
                    Q(m, kpiv), ldqm,
                    Q(m, k),    ldqm,
                    A(j, k),    ldaj,
                    T2(j, k),   T.mb,
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
