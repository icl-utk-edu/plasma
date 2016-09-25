/**
 *
 * @file
 *
 *  PLASMA is a software package provided by:
 *  University of Tennessee, US,
 *  University of Manchester, UK.
 *
 * @precisions normal z -> c d s
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

/***************************************************************************//**
 * Parallel tile triangular solve.
 * @see plasma_omp_ztrsm
 ******************************************************************************/
void plasma_pztrsm(plasma_enum_t side, plasma_enum_t uplo,
                   plasma_enum_t trans, plasma_enum_t diag,
                   plasma_complex64_t alpha, plasma_desc_t A,
                                             plasma_desc_t B,
                   plasma_sequence_t *sequence, plasma_request_t *request)
{
    int k, m, n;
    int ldak, ldam, ldan, ldbk, ldbm;
    int tempkm, tempkn, tempmm, tempnn;

    plasma_complex64_t lalpha;

    // Check sequence status.
    if (sequence->status != PlasmaSuccess) {
        plasma_request_fail(sequence, request, PlasmaErrorSequence);
        return;
    }

    if (side == PlasmaLeft) {
        if (uplo == PlasmaUpper) {
            //===========================================
            // PlasmaLeft / PlasmaUpper / PlasmaNoTrans
            //===========================================
            if (trans == PlasmaNoTrans) {
                for (k = 0; k < B.mt; k++) {
                    tempkm = plasma_tile_mdim(B, B.mt-k-1);
                    ldak   = plasma_tile_mdim(A, B.mt-k-1);
                    ldbk   = plasma_tile_mdim(B, B.mt-k-1);
                    lalpha = k == 0 ? alpha : 1.0;
                    for (n = 0; n < B.nt; n++) {
                        tempnn = plasma_tile_ndim(B, n);
                        core_omp_ztrsm(
                            side, uplo, trans, diag,
                            tempkm, tempnn,
                            lalpha, A(B.mt-k-1, B.mt-k-1), ldak,
                                    B(B.mt-k-1, n       ), ldbk);
                    }
                    for (m = k+1; m < B.mt; m++) {
                        ldam = plasma_tile_mdim(A, B.mt-1-m);
                        ldbm = plasma_tile_mdim(B, B.mt-1-m);
                        for (n = 0; n < B.nt; n++) {
                            tempnn = plasma_tile_ndim(B, n);
                            core_omp_zgemm(
                                PlasmaNoTrans, PlasmaNoTrans,
                                B.mb, tempnn, tempkm,
                                -1.0,   A(B.mt-1-m, B.mt-k-1), ldam,
                                        B(B.mt-k-1, n       ), ldbk,
                                lalpha, B(B.mt-1-m, n       ), ldbm);
                        }
                    }
                }
            }
            //================================================
            // PlasmaLeft / PlasmaUpper / Plasma[_Conj]Trans
            //================================================
            else {
                for (k = 0; k < B.mt; k++) {
                    tempkm = plasma_tile_mdim(B, k);
                    ldak   = plasma_tile_mdim(A, k);
                    ldbk   = plasma_tile_mdim(B, k);
                    lalpha = k == 0 ? alpha : 1.0;
                    for (n = 0; n < B.nt; n++) {
                        tempnn = plasma_tile_ndim(B, n);
                        core_omp_ztrsm(
                            side, uplo, trans, diag,
                            tempkm, tempnn,
                            lalpha, A(k, k), ldak,
                                    B(k, n), ldbk);
                    }
                    for (m = k+1; m < B.mt; m++) {
                        tempmm = plasma_tile_mdim(B, m);
                        ldbm   = plasma_tile_mdim(B, m);
                        for (n = 0; n < B.nt; n++) {
                            tempnn = plasma_tile_ndim(B, n);
                            core_omp_zgemm(
                                trans, PlasmaNoTrans,
                                tempmm, tempnn, B.mb,
                                -1.0,   A(k, m), ldak,
                                        B(k, n), ldbk,
                                lalpha, B(m, n), ldbm);
                        }
                    }
                }
            }
        }
        else {
            //===========================================
            // PlasmaLeft / PlasmaLower / PlasmaNoTrans
            //===========================================
            if (trans == PlasmaNoTrans) {
                for (k = 0; k < B.mt; k++) {
                    tempkm = plasma_tile_mdim(B, k);
                    ldak   = plasma_tile_mdim(A, k);
                    ldbk   = plasma_tile_mdim(B, k);
                    lalpha = k == 0 ? alpha : 1.0;
                    for (n = 0; n < B.nt; n++) {
                        tempnn = plasma_tile_ndim(B, n);
                        core_omp_ztrsm(
                            side, uplo, trans, diag,
                            tempkm, tempnn,
                            lalpha, A(k, k), ldak,
                                    B(k, n), ldbk);
                    }
                    for (m = k+1; m < B.mt; m++) {
                        tempmm = plasma_tile_mdim(B, m);
                        ldam   = plasma_tile_mdim(A, m);
                        ldbm   = plasma_tile_mdim(B, m);
                        for (n = 0; n < B.nt; n++) {
                            tempnn = plasma_tile_ndim(B, n);
                            core_omp_zgemm(
                                PlasmaNoTrans, PlasmaNoTrans,
                                tempmm, tempnn, B.mb,
                                -1.0,   A(m, k), ldam,
                                        B(k, n), ldbk,
                                lalpha, B(m, n), ldbm);
                        }
                    }
                }
            }
            //================================================
            // PlasmaLeft / PlasmaLower / Plasma[_Conj]Trans
            //================================================
            else {
                for (k = 0; k < B.mt; k++) {
                    tempkm = plasma_tile_mdim(B, B.mt-k-1);
                    ldak   = plasma_tile_mdim(A, B.mt-k-1);
                    ldbk   = plasma_tile_mdim(B, B.mt-k-1);
                    lalpha = k == 0 ? alpha : 1.0;
                    for (n = 0; n < B.nt; n++) {
                        tempnn = plasma_tile_ndim(B, n);
                        core_omp_ztrsm(
                            side, uplo, trans, diag,
                            tempkm, tempnn,
                            lalpha, A(B.mt-k-1, B.mt-k-1), ldak,
                                    B(B.mt-k-1, n       ), ldbk);
                    }
                    for (m = k+1; m < B.mt; m++) {
                        tempmm = plasma_tile_mdim(B, m);
                        ldbm   = plasma_tile_mdim(B, B.mt-1-m);
                        for (n = 0; n < B.nt; n++) {
                            tempnn = plasma_tile_ndim(B, n);
                            core_omp_zgemm(
                                trans, PlasmaNoTrans,
                                B.mb, tempnn, tempkm,
                                -1.0,   A(B.mt-k-1, B.mt-1-m), ldak,
                                        B(B.mt-k-1, n       ), ldbk,
                                lalpha, B(B.mt-1-m, n       ), ldbm);
                        }
                    }
                }
            }
        }
    }
    else {
        if (uplo == PlasmaUpper) {
            //============================================
            // PlasmaRight / PlasmaUpper / PlasmaNoTrans
            //============================================
            if (trans == PlasmaNoTrans) {
                for (k = 0; k < B.nt; k++) {
                    tempkn = plasma_tile_ndim(B, k);
                    ldak   = plasma_tile_mdim(A, k);
                    lalpha = k == 0 ? alpha : 1.0;
                    for (m = 0; m < B.mt; m++) {
                        tempmm = plasma_tile_mdim(B, m);
                        ldbm   = plasma_tile_mdim(B, m);
                        core_omp_ztrsm(
                            side, uplo, trans, diag,
                            tempmm, tempkn,
                            lalpha, A(k, k), ldak,
                                    B(m, k), ldbm);
                    }
                    for (m = 0; m < B.mt; m++) {
                        tempmm = plasma_tile_mdim(B, m);
                        ldbm   = plasma_tile_mdim(B, m);
                        for (n = k+1; n < B.nt; n++) {
                            tempnn = plasma_tile_ndim(B, n);
                            core_omp_zgemm(
                                PlasmaNoTrans, PlasmaNoTrans,
                                tempmm, tempnn, B.mb,
                                -1.0,   B(m, k), ldbm,
                                        A(k, n), ldak,
                                lalpha, B(m, n), ldbm);
                        }
                    }
                }
            }
            //=================================================
            // PlasmaRight / PlasmaUpper / Plasma[_Conj]Trans
            //=================================================
            else {
                for (k = 0; k < B.nt; k++) {
                    tempkn = plasma_tile_ndim(B, B.nt-k-1);
                    ldak   = plasma_tile_mdim(A, B.nt-k-1);
                    for (m = 0; m < B.mt; m++) {
                        tempmm = plasma_tile_mdim(B, m);
                        ldbm   = plasma_tile_mdim(B, m);
                        core_omp_ztrsm(
                            side, uplo, trans, diag,
                            tempmm, tempkn,
                            alpha, A(B.nt-k-1, B.nt-k-1), ldak,
                                   B(m,        B.nt-k-1), ldbm);

                        for (n = k+1; n < B.nt; n++) {
                            ldan = plasma_tile_mdim(A, B.nt-1-n);
                            core_omp_zgemm(
                                PlasmaNoTrans, trans,
                                tempmm, B.nb, tempkn,
                                -1.0/alpha, B(m,        B.nt-k-1), ldbm,
                                            A(B.nt-1-n, B.nt-k-1), ldan,
                                1.0,        B(m,        B.nt-1-n), ldbm);
                        }
                    }
                }
            }
        }
        else {
            //============================================
            // PlasmaRight / PlasmaLower / PlasmaNoTrans
            //============================================
            if (trans == PlasmaNoTrans) {
                for (k = 0; k < B.nt; k++) {
                    tempkn = plasma_tile_ndim(B, B.nt-k-1);
                    ldak   = plasma_tile_mdim(A, B.nt-k-1);
                    lalpha = k == 0 ? alpha : 1.0;
                    for (m = 0; m < B.mt; m++) {
                        tempmm = plasma_tile_mdim(B, m);
                        ldbm   = plasma_tile_mdim(B, m);
                        core_omp_ztrsm(
                            side, uplo, trans, diag,
                            tempmm, tempkn,
                            lalpha, A(B.nt-k-1, B.nt-k-1), ldak,
                                    B(m,        B.nt-k-1), ldbm);

                        for (n = k+1; n < B.nt; n++) {
                            core_omp_zgemm(
                                PlasmaNoTrans, PlasmaNoTrans,
                                tempmm, B.nb, tempkn,
                                -1.0,   B(m,        B.nt-k-1), ldbm,
                                        A(B.nt-1-k, B.nt-1-n), ldak,
                                lalpha, B(m,        B.nt-1-n), ldbm);
                        }
                    }
                }
            }
            //==================================================
            //  PlasmaRight / PlasmaLower / Plasma[_Conj]Trans
            //==================================================
            else {
                for (k = 0; k < B.nt; k++) {
                    tempkn = plasma_tile_ndim(B, k);
                    ldak   = plasma_tile_mdim(A, k);
                    for (m = 0; m < B.mt; m++) {
                        tempmm = plasma_tile_mdim(B, m);
                        ldbm   = plasma_tile_mdim(B, m);
                        core_omp_ztrsm(
                            side, uplo, trans, diag,
                            tempmm, tempkn,
                            alpha, A(k, k), ldak,
                                   B(m, k), ldbm);

                        for (n = k+1; n < B.nt; n++) {
                            tempnn = plasma_tile_ndim(B, n);
                            ldan   = plasma_tile_mdim(A, n);
                            core_omp_zgemm(
                                PlasmaNoTrans, trans,
                                tempmm, tempnn, B.mb,
                                -1.0/alpha, B(m, k), ldbm,
                                            A(n, k), ldan,
                                1.0,        B(m, n), ldbm);
                        }
                    }
                }
            }
        }
    }
}
