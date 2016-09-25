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
 *  Parallel application of Q using tile V - LQ factorization
 * @see plasma_omp_zgelqs
 **/
void plasma_pzunmlq(plasma_enum_t side, plasma_enum_t trans,
                    plasma_desc_t A, plasma_desc_t B, plasma_desc_t T,
                    plasma_workspace_t *work,
                    plasma_sequence_t *sequence, plasma_request_t *request)
{
    int k, m, n;
    int ldak, ldbk, ldbm;
    int tempmm, tempnn, tempkn, tempkm, tempkmin;
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
        //=============================
        // PlasmaLeft / PlasmaNoTrans
        //=============================
        if (trans == PlasmaNoTrans) {
            for (k = 0; k < minMT; k++) {
                tempkm   = plasma_tile_mdim(B, k);
                tempkmin = k == minMT-1 ? minM-k*A.nb : A.nb;
                ldak = plasma_tile_mdim(A, k);
                ldbk = plasma_tile_mdim(B, k);
                for (n = 0; n < B.nt; n++) {
                    tempnn = plasma_tile_ndim(B, n);
                    core_omp_zunmlq(
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
                    ldbm   = plasma_tile_mdim(B, m);
                    for (n = 0; n < B.nt; n++) {
                        tempnn = plasma_tile_ndim(B, n);
                        core_omp_ztsmlq(
                                side, trans,
                                B.mb, tempnn, tempmm, tempnn, tempkmin,
                                ib, T.nb,
                                B(k, n), ldbk,
                                B(m, n), ldbm,
                                A(k, m), ldak,
                                T(k, m), T.mb,
                                work,
                                sequence, request);
                    }
                }
            }
        }
        //==================================
        // PlasmaLeft / Plasma[_Conj]Trans
        //==================================
        else {
            for (k = minMT-1; k >= 0; k--) {
                tempkm   = plasma_tile_mdim(B, k);
                tempkmin = k == minMT-1 ? minM-k*A.nb : A.nb;
                ldak = plasma_tile_mdim(A, k);
                ldbk = plasma_tile_mdim(B, k);
                for (m = B.mt-1; m > k; m--) {
                    tempmm = plasma_tile_mdim(B, m);
                    ldbm   = plasma_tile_mdim(B, m);
                    for (n = 0; n < B.nt; n++) {
                        tempnn = plasma_tile_ndim(B, n);
                        core_omp_ztsmlq(
                                side, trans,
                                B.mb, tempnn, tempmm, tempnn, tempkmin,
                                ib, T.nb,
                                B(k, n), ldbk,
                                B(m, n), ldbm,
                                A(k, m), ldak,
                                T(k, m), T.mb,
                                work,
                                sequence, request);
                    }
                }
                for (n = 0; n < B.nt; n++) {
                    tempnn = plasma_tile_ndim(B, n);
                    core_omp_zunmlq(
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
        //==============================
        // PlasmaRight / PlasmaNoTrans
        //==============================
        if (trans == PlasmaNoTrans) {
            for (k = minMT-1; k >= 0; k--) {
                tempkn   = plasma_tile_ndim(B, k);
                tempkmin = k == minMT-1 ? minM-k*A.nb : A.nb;
                ldak = plasma_tile_mdim(A, k);
                for (n = B.nt-1; n > k; n--) {
                    tempnn = plasma_tile_ndim(B, n);
                    for (m = 0; m < B.mt; m++) {
                        tempmm = plasma_tile_mdim(B, m);
                        ldbm   = plasma_tile_mdim(B, m);
                        core_omp_ztsmlq(
                                side, trans,
                                tempmm, B.nb, tempmm, tempnn, tempkmin,
                                ib, T.nb,
                                B(m, k), ldbm,
                                B(m, n), ldbm,
                                A(k, n), ldak,
                                T(k, n), T.mb,
                                work,
                                sequence, request);
                    }
                }
                for (m = 0; m < B.mt; m++) {
                    tempmm = plasma_tile_mdim(B, m);
                    ldbm   = plasma_tile_mdim(B, m);
                    core_omp_zunmlq(
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
        //===================================
        // PlasmaRight / Plasma[_Conj]Trans
        //===================================
        else {
            for (k = 0; k < minMT; k++) {
                tempkn   = plasma_tile_ndim(B, k);
                tempkmin = k == minMT-1 ? minM-k*A.mb : A.mb;
                ldak = plasma_tile_mdim(A, k);
                for (m = 0; m < B.mt; m++) {
                    tempmm = plasma_tile_mdim(B, m);
                    ldbm   = plasma_tile_mdim(B, m);
                    core_omp_zunmlq(
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
                    for (m = 0; m < B.mt; m++) {
                        tempmm = plasma_tile_mdim(B, m);
                        ldbm   = plasma_tile_mdim(B, m);
                        core_omp_ztsmlq(
                                side, trans,
                                tempmm, B.nb, tempmm, tempnn, tempkmin,
                                ib, T.nb,
                                B(m, k), ldbm,
                                B(m, n), ldbm,
                                A(k, n), ldak,
                                T(k, n), T.mb,
                                work,
                                sequence, request);
                    }
                }
            }
        }
    }
}
