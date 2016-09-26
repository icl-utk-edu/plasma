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
#define B(m, n) (plasma_complex64_t*)plasma_tile_addr(B, m, n)
#define T(m, n) (plasma_complex64_t*)plasma_tile_addr(T, m, n)

/***************************************************************************//**
 *  Parallel application of Q using tile V - QR factorization
 * @see plasma_omp_zgeqrs
 **/
void plasma_pzunmqr(plasma_enum_t side, plasma_enum_t trans,
                    plasma_desc_t A, plasma_desc_t B, plasma_desc_t T,
                    plasma_workspace_t *work,
                    plasma_sequence_t *sequence, plasma_request_t *request)
{
    int k, m, n;
    int ldak, ldbk, ldam, ldan, ldbm;
    int tempkm, tempnn, tempkmin, tempmm, tempkn;
    int minMT, minM;

    // Check sequence status.
    if (sequence->status != PlasmaSuccess) {
        plasma_request_fail(sequence, request, PlasmaErrorSequence);
        return;
    }

    // Set inner blocking from the T tile row-dimension.
    int ib = T.mb;

    if (A.m > A.n) {
      minM  = A.n;
      minMT = A.nt;
    }
    else {
      minM  = A.m;
      minMT = A.mt;
    }

    if (side == PlasmaLeft) {
        //================================
        // PlasmaLeft / Plasma_ConjTrans
        //================================
        if (trans == Plasma_ConjTrans) {
            for (k = 0; k < minMT; k++) {
                tempkm   = plasma_tile_mdim(B, k);
                tempkmin = k == minMT-1 ? minM-k*A.nb : A.nb;
                ldak = plasma_tile_mdim(A, k);
                ldbk = plasma_tile_mdim(B, k);
                for (n = 0; n < B.nt; n++) {
                    tempnn = plasma_tile_ndim(B, n);
                    core_omp_zunmqr(
                        side, trans,
                        tempkm, tempnn, tempkmin, ib, T.nb,
                        A(k, k), ldak,
                        T(k, k), T.mb,
                        B(k, n), ldbk,
                        work,
                        sequence, request);
                }
                for (m = k+1; m < B.mt; m++) {
                    tempmm = plasma_tile_mdim(B, m);
                    ldam   = plasma_tile_mdim(A, m);
                    ldbm   = plasma_tile_mdim(B, m);
                    for (n = 0; n < B.nt; n++) {
                        tempnn = plasma_tile_ndim(B, n);
                        core_omp_ztsmqr(
                            side, trans,
                            B.mb, tempnn, tempmm, tempnn, tempkmin, ib, T.nb,
                            B(k, n), ldbk,
                            B(m, n), ldbm,
                            A(m, k), ldam,
                            T(m, k), T.mb,
                            work,
                            sequence, request);
                    }
                }
            }
        }
        //=============================
        // PlasmaLeft / PlasmaNoTrans
        //=============================
        else {
            for (k = minMT-1; k >= 0; k--) {
                tempkm = plasma_tile_mdim(B, k);
                tempkmin = k == minMT-1 ? minM-k*A.nb : A.nb;
                ldak = plasma_tile_mdim(A, k);
                ldbk = plasma_tile_mdim(B, k);
                for (m = B.mt-1; m > k; m--) {
                    tempmm = plasma_tile_mdim(B, m);
                    ldam   = plasma_tile_mdim(A, m);
                    ldbm   = plasma_tile_mdim(B, m);
                    for (n = 0; n < B.nt; n++) {
                        tempnn = plasma_tile_ndim(B, n);
                        core_omp_ztsmqr(
                            side, trans,
                            B.mb, tempnn, tempmm, tempnn, tempkmin, ib, T.nb,
                            B(k, n), ldbk,
                            B(m, n), ldbm,
                            A(m, k), ldam,
                            T(m, k), T.mb,
                            work,
                            sequence, request);
                    }
                }
                for (n = 0; n < B.nt; n++) {
                    tempnn = plasma_tile_ndim(B, n);
                    core_omp_zunmqr(
                        side, trans,
                        tempkm, tempnn, tempkmin, ib, T.nb,
                        A(k, k), ldak,
                        T(k, k), T.mb,
                        B(k, n), ldbk,
                        work,
                        sequence, request);
                }
            }
        }
    }
    else {
        //=================================
        // PlasmaRight / Plasma_ConjTrans
        //=================================
        if (trans == Plasma_ConjTrans) {
            for (k = minMT-1; k >= 0; k--) {
                tempkn = plasma_tile_ndim(B, k);
                tempkmin = k == minMT-1 ? minM-k*A.nb : A.nb;
                ldak = plasma_tile_mdim(A, k);
                ldbk = plasma_tile_mdim(B, k);
                for (n = B.nt-1; n > k; n--) {
                    tempnn = plasma_tile_ndim(B, n);
                    ldan   = plasma_tile_mdim(A, n);
                    for (m = 0; m < B.mt; m++) {
                        tempmm = plasma_tile_mdim(B, m);
                        ldbm   = plasma_tile_mdim(B, m);
                        core_omp_ztsmqr(
                            side, trans,
                            tempmm, B.nb, tempmm, tempnn, tempkmin, ib, T.nb,
                            B(m, k), ldbm,
                            B(m, n), ldbm,
                            A(n, k), ldan,
                            T(n, k), T.mb,
                            work,
                            sequence, request);
                    }
                }
                for (m = 0; m < B.mt; m++) {
                    tempmm = plasma_tile_mdim(B, m);
                    ldbm   = plasma_tile_mdim(B, m);
                    core_omp_zunmqr(
                        side, trans,
                        tempmm, tempkn, tempkmin, ib, T.nb,
                        A(k, k), ldak,
                        T(k, k), T.mb,
                        B(m, k), ldbm,
                        work,
                        sequence, request);
                }
            }
        }
        //==============================
        // PlasmaRight / PlasmaNoTrans
        //==============================
        else {
            for (k = 0; k < minMT; k++) {
                tempkn   = plasma_tile_ndim(B, k);
                tempkmin = k == minMT-1 ? minM-k*A.nb : A.nb;
                ldak = plasma_tile_mdim(A, k);
                for (m = 0; m < B.mt; m++) {
                    tempmm = plasma_tile_mdim(B, m);
                    ldbm   = plasma_tile_mdim(B, m);
                    core_omp_zunmqr(
                        side, trans,
                        tempmm, tempkn, tempkmin, ib, T.nb,
                        A(k, k), ldak,
                        T(k, k), T.mb,
                        B(m, k), ldbm,
                        work,
                        sequence, request);
                }
                for (n = k+1; n < B.nt; n++) {
                    tempnn = plasma_tile_ndim(B, n);
                    ldan   = plasma_tile_mdim(A, n);
                    for (m = 0; m < B.mt; m++) {
                        tempmm = plasma_tile_mdim(B, m);
                        ldbm   = plasma_tile_mdim(B, m);
                        core_omp_ztsmqr(
                            side, trans,
                            tempmm, B.nb, tempmm, tempnn, tempkmin, ib, T.nb,
                            B(m, k), ldbm,
                            B(m, n), ldbm,
                            A(n, k), ldan,
                            T(n, k), T.mb,
                            work,
                            sequence, request);
                    }
                }
            }
        }
    }
}
