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

    // Check sequence status.
    if (sequence->status != PlasmaSuccess) {
        plasma_request_fail(sequence, request, PlasmaErrorSequence);
        return;
    }

    // Precompute order of QR operations.
    int *operations = NULL;
    int noperations;
    plasma_rh_tree_operations(A.mt, A.nt, &operations, &noperations);

    //printf("noperations %d\n",noperations);
    //int j,k,kpiv;
    //for (int iop = 0; iop < noperations; iop++) {
    //    plasma_qr_operation_get(operations, iop, &j, &k, &kpiv);
    //    if (omp_get_thread_num() == 0) {
    //        printf(" %d, %d, %d \n", j,k,kpiv);
    //    }
    //}

    // Set inner blocking from the T tile row-dimension.
    int ib = T.mb;

    for (int iop = 0; iop < noperations; iop++) {

        int j, k, kpiv;
        plasma_rh_tree_operation_get(operations, iop, &j, &k, &kpiv);

        int nvaj    = plasma_tile_nview(A, j);
        int mvak    = plasma_tile_mview(A, k);
        int ldak    = plasma_tile_mmain(A, k);

        if (kpiv < 0) {
            // triangularization
            // GEQRT(k,j)
            if (debug) printf("GEQRT (%d,%d,%d,%d) ", k, j, mvak, nvaj);
            core_omp_zgeqrt(
                mvak, nvaj, ib,
                A(k, j), ldak,
                T(k, j), T.mb,
                work,
                sequence, request);

            for (int jj = j + 1; jj < A.nt; jj++) {
                int nvajj = plasma_tile_nview(A, jj);

                // UNMQR(k,j,jj)
                if (debug) printf("UNMQR (%d,%d,%d,%d,%d,%d) ", k, j, jj, mvak, nvajj, imin(nvaj, mvak));
                core_omp_zunmqr(
                    PlasmaLeft, Plasma_ConjTrans,
                    mvak, nvajj, imin(nvaj, mvak), ib,
                    A(k, j), ldak,
                    T(k, j), T.mb,
                    A(k, jj), ldak,
                    work,
                    sequence, request);
            }

            if (debug) printf("\n ");
        }
        else {
            // elimination of the tile
            int mvakpiv = plasma_tile_mview(A, kpiv);
            int ldakpiv = plasma_tile_mmain(A, kpiv);

            // TTQRT(A.mt- kk - 1, pivpmkk, j)
            if (debug) printf("TTQRT (%d,%d,%d,%d,%d) ", k, kpiv, j, mvak, nvaj);
            core_omp_zttqrt(
                mvak, nvaj, ib,
                A(kpiv, j), ldakpiv,
                A(k,  j),   ldak,
                T2(k, j),   T.mb,
                work,
                sequence, request);

            for (int jj = j + 1; jj < A.nt; jj++) {
                int nvajj = plasma_tile_nview(A, jj);

                // TTMQR(A.mt-kk-1,pivpmkk,j,jj)
                if (debug) printf("TTMQR (%d,%d,%d,%d,%d,%d,%d,%d,%d)) ", k, kpiv,
                                  j, jj, mvakpiv, nvajj, mvak, nvajj, nvaj);
                core_omp_zttmqr(
                    PlasmaLeft, Plasma_ConjTrans,
                    mvakpiv, nvajj, mvak, nvajj, nvaj, ib,
                    A(kpiv, jj), ldakpiv,
                    A(k,    jj), ldak,
                    A(k,    j),  ldak,
                    T2(k,   j),  T.mb,
                    work,
                    sequence, request);
            }

            if (debug) printf("\n ");

            if (debug) printf(" ==== \n ");
        }
    }

    free(operations);
}
