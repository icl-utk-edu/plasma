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
#include "plasma_types.h"
#include "plasma_internal.h"
#include "plasma_rh_tree.h"
#include "core_blas_z.h"

#define A(m, n) (plasma_complex64_t*)plasma_tile_addr(A, m, n)
#define T(m, n) (plasma_complex64_t*)plasma_tile_addr(T, m, n)
#define T2(m, n) (plasma_complex64_t*)plasma_tile_addr(T, m, n+(T.nt/2))
/***************************************************************************//**
 *  Parallel tile QR factorization based on a tree Householder reduction
 * @see plasma_omp_zgeqrf
 **/
void plasma_pzgeqrfrh(plasma_desc_t A, plasma_desc_t T,
                      plasma_workspace_t work,
                      plasma_sequence_t *sequence, plasma_request_t *request)
{
    static const int debug = 0;
    if (debug) printf("executing pzgeqrfrh\n");

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

    for (int iop = 0; iop < noperations; iop++) {

        int j, k, kpiv;
        plasma_enum_t kernel;
        plasma_rh_tree_operation_get(operations, iop, &kernel, &j, &k, &kpiv);

        int nvaj    = plasma_tile_nview(A, j);
        int mvak    = plasma_tile_mview(A, k);
        int ldak    = plasma_tile_mmain(A, k);

        if (kernel == PlasmaGEKernel) {
            // triangularization
            if (debug) printf("GEQRT (%d,%d) ", k, j);
            core_omp_zgeqrt(
                mvak, nvaj, ib,
                A(k, j), ldak,
                T(k, j), T.mb,
                work,
                sequence, request);

            for (int jj = j + 1; jj < A.nt; jj++) {
                int nvajj = plasma_tile_nview(A, jj);

                if (debug) printf("UNMQR (%d,%d,%d) ", k, j, jj);
                core_omp_zunmqr(
                    PlasmaLeft, Plasma_ConjTrans,
                    mvak, nvajj, imin(mvak, nvaj), ib,
                    A(k, j), ldak,
                    T(k, j), T.mb,
                    A(k, jj), ldak,
                    work,
                    sequence, request);
            }

            if (debug) printf("\n ");
        }
        else if (kernel == PlasmaTTKernel) {
            // elimination of the tile
            int mvakpiv = plasma_tile_mview(A, kpiv);
            int ldakpiv = plasma_tile_mmain(A, kpiv);

            if (debug) printf("TTQRT (%d,%d,%d) ", k, kpiv, j);
            core_omp_zttqrt(
                mvak, nvaj, ib,
                A(kpiv, j), ldakpiv,
                A(k,  j),   ldak,
                T2(k, j),   T.mb,
                work,
                sequence, request);

            for (int jj = j + 1; jj < A.nt; jj++) {
                int nvajj = plasma_tile_nview(A, jj);

                if (debug) printf("TTMQR (%d,%d,%d,%d)) ", k, kpiv, j, jj);
                core_omp_zttmqr(
                    PlasmaLeft, Plasma_ConjTrans,
                    mvakpiv, nvajj, mvak, nvajj, imin(mvakpiv+mvak, nvaj), ib,
                    A(kpiv, jj), ldakpiv,
                    A(k,    jj), ldak,
                    A(k,    j),  ldak,
                    T2(k,   j),  T.mb,
                    work,
                    sequence, request);
            }

            if (debug) printf("\n ");
        }
        else if (kernel == PlasmaTSKernel) {
            // elimination of the tile
            int mvakpiv = plasma_tile_mview(A, kpiv);
            int ldakpiv = plasma_tile_mmain(A, kpiv);

            if (debug) printf("TSQRT (%d,%d,%d) ", k, kpiv, j);
            core_omp_ztsqrt(
                mvak, nvaj, ib,
                A(kpiv, j), ldakpiv,
                A(k,  j),   ldak,
                T2(k, j),   T.mb,
                work,
                sequence, request);

            for (int jj = j + 1; jj < A.nt; jj++) {
                int nvajj = plasma_tile_nview(A, jj);

                if (debug) printf("TSMQR (%d,%d,%d,%d)) ", k, kpiv, j, jj);
                core_omp_ztsmqr(
                    PlasmaLeft, Plasma_ConjTrans,
                    mvakpiv, nvajj, mvak, nvajj, imin(mvakpiv+mvak, nvaj), ib,
                    A(kpiv, jj), ldakpiv,
                    A(k,    jj), ldak,
                    A(k,    j),  ldak,
                    T2(k,   j),  T.mb,
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
