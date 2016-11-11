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
 *  Parallel tile LQ factorization based on a tree Householder reduction
 * @see plasma_omp_zgelqf
 **/
void plasma_pzgelqfrh(plasma_desc_t A, plasma_desc_t T,
                      plasma_workspace_t work,
                      plasma_sequence_t *sequence, plasma_request_t *request)
{
    static const int debug = 0;
    if (debug) printf("executing pzgelqfrh\n");

    // Check sequence status.
    if (sequence->status != PlasmaSuccess) {
        plasma_request_fail(sequence, request, PlasmaErrorSequence);
        return;
    }

    // Precompute order of LQ operations - compute it as for QR
    // and transpose it.
    int *operations = NULL;
    int noperations;
    // Transpose m and n to reuse the QR tree.
    plasma_rh_tree_operations(A.nt, A.mt, &operations, &noperations);

    // Set inner blocking from the T tile row-dimension.
    int ib = T.mb;

    for (int iop = 0; iop < noperations; iop++) {
        int j, k, kpiv;
        plasma_enum_t kernel;
        // j is row, k and kpiv are columns
        plasma_rh_tree_operation_get(operations, iop, &kernel, &j, &k, &kpiv);

        int mvaj    = plasma_tile_mview(A, j);
        int nvak    = plasma_tile_nview(A, k);
        int ldaj    = plasma_tile_mmain(A, j);

        if (kernel == PlasmaGEKernel) {
            // triangularization
            if (debug) printf("GELQT (%d,%d) ", j, k);
            core_omp_zgelqt(
                mvaj, nvak, ib,
                A(j, k), ldaj,
                T(j, k), T.mb,
                work,
                sequence, request);

            for (int jj = j + 1; jj < A.mt; jj++) {
                int mvajj = plasma_tile_mview(A, jj);
                int ldajj = plasma_tile_mmain(A, jj);

                if (debug) printf("UNMLQ (%d,%d,%d) ", j, jj, k);
                core_omp_zunmlq(
                    PlasmaRight, Plasma_ConjTrans,
                    mvajj, nvak, imin(mvaj, nvak), ib,
                    A(j,  k), ldaj,
                    T(j,  k), T.mb,
                    A(jj, k), ldajj,
                    work,
                    sequence, request);
            }

            if (debug) printf("\n ");
        }
        else if (kernel == PlasmaTTKernel) {
            // elimination of the tile
            int nvakpiv = plasma_tile_nview(A, kpiv);

            if (debug) printf("TTLQT (%d,%d,%d) ", j, k, kpiv);
            core_omp_zttlqt(
                mvaj, nvak, ib,
                A(j,  kpiv), ldaj,
                A(j,  k),    ldaj,
                T2(j, k),    T.mb,
                work,
                sequence, request);

            for (int jj = j + 1; jj < A.mt; jj++) {
                int mvajj = plasma_tile_mview(A, jj);
                int ldajj = plasma_tile_mmain(A, jj);

                if (debug) printf("TTMLQ (%d,%d,%d,%d)) ", j, jj, k, kpiv);
                core_omp_zttmlq(
                    PlasmaRight, Plasma_ConjTrans,
                    mvajj, nvakpiv, mvajj, nvak, imin(mvaj, nvakpiv+nvak), ib,
                    A(jj, kpiv), ldajj,
                    A(jj, k),    ldajj,
                    A(j,  k),    ldaj,
                    T2(j, k),    T.mb,
                    work,
                    sequence, request);
            }

            if (debug) printf("\n ");
        }
        else if (kernel == PlasmaTSKernel) {
            // elimination of the tile
            int nvakpiv = plasma_tile_nview(A, kpiv);

            if (debug) printf("TSLQT (%d,%d,%d) ", j, k, kpiv);
            core_omp_ztslqt(
                mvaj, nvak, ib,
                A(j,  kpiv), ldaj,
                A(j,  k),    ldaj,
                T2(j, k),    T.mb,
                work,
                sequence, request);

            for (int jj = j + 1; jj < A.mt; jj++) {
                int mvajj = plasma_tile_mview(A, jj);
                int ldajj = plasma_tile_mmain(A, jj);

                if (debug) printf("TSMLQ (%d,%d,%d,%d)) ", j, jj, k, kpiv);
                core_omp_ztsmlq(
                    PlasmaRight, Plasma_ConjTrans,
                    mvajj, nvakpiv, mvajj, nvak, imin(mvaj, nvakpiv+nvak), ib,
                    A(jj, kpiv), ldajj,
                    A(jj, k),    ldajj,
                    A(j,  k),    ldaj,
                    T2(j, k),    T.mb,
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
