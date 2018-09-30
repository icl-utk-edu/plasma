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
#include <plasma_core_blas.h>

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
    // Return if failed sequence.
    if (sequence->status != PlasmaSuccess)
        return;

    if (side == PlasmaLeft) {
        if (uplo == PlasmaUpper) {
            //===========================================
            // PlasmaLeft / PlasmaUpper / PlasmaNoTrans
            //===========================================
            if (trans == PlasmaNoTrans) {
                for (int k = 0; k < B.mt; k++) {
                    int mvbk = plasma_tile_mview(B, B.mt-k-1);
                    int ldak = plasma_tile_mmain(A, B.mt-k-1);
                    int ldbk = plasma_tile_mmain(B, B.mt-k-1);
                    plasma_complex64_t lalpha = k == 0 ? alpha : 1.0;
                    for (int n = 0; n < B.nt; n++) {
                        int nvbn = plasma_tile_nview(B, n);
                        plasma_core_omp_ztrsm(
                            side, uplo, trans, diag,
                            mvbk, nvbn,
                            lalpha, A(B.mt-k-1, B.mt-k-1), ldak,
                                    B(B.mt-k-1, n       ), ldbk,
                            sequence, request);
                    }
                    for (int m = k+1; m < B.mt; m++) {
                        int ldam = plasma_tile_mmain(A, B.mt-1-m);
                        int ldbm = plasma_tile_mmain(B, B.mt-1-m);
                        for (int n = 0; n < B.nt; n++) {
                            int nvbn = plasma_tile_nview(B, n);
                            plasma_core_omp_zgemm(
                                PlasmaNoTrans, PlasmaNoTrans,
                                B.mb, nvbn, mvbk,
                                -1.0,   A(B.mt-1-m, B.mt-k-1), ldam,
                                        B(B.mt-k-1, n       ), ldbk,
                                lalpha, B(B.mt-1-m, n       ), ldbm,
                                sequence, request);
                        }
                    }
                }
            }
            //================================================
            // PlasmaLeft / PlasmaUpper / Plasma[_Conj]Trans
            //================================================
            else {
                for (int k = 0; k < B.mt; k++) {
                    int mvbk = plasma_tile_mview(B, k);
                    int ldak = plasma_tile_mmain(A, k);
                    int ldbk = plasma_tile_mmain(B, k);
                    plasma_complex64_t lalpha = k == 0 ? alpha : 1.0;
                    for (int n = 0; n < B.nt; n++) {
                        int nvbn = plasma_tile_nview(B, n);
                        plasma_core_omp_ztrsm(
                            side, uplo, trans, diag,
                            mvbk, nvbn,
                            lalpha, A(k, k), ldak,
                                    B(k, n), ldbk,
                            sequence, request);
                    }
                    for (int m = k+1; m < B.mt; m++) {
                        int mvbm = plasma_tile_mview(B, m);
                        int ldbm = plasma_tile_mmain(B, m);
                        for (int n = 0; n < B.nt; n++) {
                            int nvbn = plasma_tile_nview(B, n);
                            plasma_core_omp_zgemm(
                                trans, PlasmaNoTrans,
                                mvbm, nvbn, B.mb,
                                -1.0,   A(k, m), ldak,
                                        B(k, n), ldbk,
                                lalpha, B(m, n), ldbm,
                                sequence, request);
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
                for (int k = 0; k < B.mt; k++) {
                    int mvbk = plasma_tile_mview(B, k);
                    int ldak = plasma_tile_mmain(A, k);
                    int ldbk = plasma_tile_mmain(B, k);
                    plasma_complex64_t lalpha = k == 0 ? alpha : 1.0;
                    for (int n = 0; n < B.nt; n++) {
                        int nvbn = plasma_tile_nview(B, n);
                        plasma_core_omp_ztrsm(
                            side, uplo, trans, diag,
                            mvbk, nvbn,
                            lalpha, A(k, k), ldak,
                                    B(k, n), ldbk,
                            sequence, request);
                    }
                    for (int m = k+1; m < B.mt; m++) {
                        int mvbm = plasma_tile_mview(B, m);
                        int ldam = plasma_tile_mmain(A, m);
                        int ldbm = plasma_tile_mmain(B, m);
                        for (int n = 0; n < B.nt; n++) {
                            int nvbn = plasma_tile_nview(B, n);
                            plasma_core_omp_zgemm(
                                PlasmaNoTrans, PlasmaNoTrans,
                                mvbm, nvbn, B.mb,
                                -1.0,   A(m, k), ldam,
                                        B(k, n), ldbk,
                                lalpha, B(m, n), ldbm,
                                sequence, request);
                        }
                    }
                }
            }
            //================================================
            // PlasmaLeft / PlasmaLower / Plasma[_Conj]Trans
            //================================================
            else {
                for (int k = 0; k < B.mt; k++) {
                    int mvbk = plasma_tile_mview(B, B.mt-k-1);
                    int ldak = plasma_tile_mmain(A, B.mt-k-1);
                    int ldbk = plasma_tile_mmain(B, B.mt-k-1);
                    plasma_complex64_t lalpha = k == 0 ? alpha : 1.0;
                    for (int n = 0; n < B.nt; n++) {
                        int nvbn = plasma_tile_nview(B, n);
                        plasma_core_omp_ztrsm(
                            side, uplo, trans, diag,
                            mvbk, nvbn,
                            lalpha, A(B.mt-k-1, B.mt-k-1), ldak,
                                    B(B.mt-k-1, n       ), ldbk,
                            sequence, request);
                    }
                    for (int m = k+1; m < B.mt; m++) {
                        int ldbm = plasma_tile_mmain(B, B.mt-1-m);
                        for (int n = 0; n < B.nt; n++) {
                            int nvbn = plasma_tile_nview(B, n);
                            plasma_core_omp_zgemm(
                                trans, PlasmaNoTrans,
                                B.mb, nvbn, mvbk,
                                -1.0,   A(B.mt-k-1, B.mt-1-m), ldak,
                                        B(B.mt-k-1, n       ), ldbk,
                                lalpha, B(B.mt-1-m, n       ), ldbm,
                                sequence, request);
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
                for (int k = 0; k < B.nt; k++) {
                    int nvbk = plasma_tile_nview(B, k);
                    int ldak = plasma_tile_mmain(A, k);
                    plasma_complex64_t lalpha = k == 0 ? alpha : 1.0;
                    for (int m = 0; m < B.mt; m++) {
                        int mvbm = plasma_tile_mview(B, m);
                        int ldbm = plasma_tile_mmain(B, m);
                        plasma_core_omp_ztrsm(
                            side, uplo, trans, diag,
                            mvbm, nvbk,
                            lalpha, A(k, k), ldak,
                                    B(m, k), ldbm,
                            sequence, request);
                    }
                    for (int m = 0; m < B.mt; m++) {
                        int mvbm = plasma_tile_mview(B, m);
                        int ldbm = plasma_tile_mmain(B, m);
                        for (int n = k+1; n < B.nt; n++) {
                            int nvbn = plasma_tile_nview(B, n);
                            plasma_core_omp_zgemm(
                                PlasmaNoTrans, PlasmaNoTrans,
                                mvbm, nvbn, B.mb,
                                -1.0,   B(m, k), ldbm,
                                        A(k, n), ldak,
                                lalpha, B(m, n), ldbm,
                                sequence, request);
                        }
                    }
                }
            }
            //=================================================
            // PlasmaRight / PlasmaUpper / Plasma[_Conj]Trans
            //=================================================
            else {
                for (int k = 0; k < B.nt; k++) {
                    int nvbk = plasma_tile_nview(B, B.nt-k-1);
                    int ldak = plasma_tile_mmain(A, B.nt-k-1);
                    for (int m = 0; m < B.mt; m++) {
                        int mvbm = plasma_tile_mview(B, m);
                        int ldbm   = plasma_tile_mmain(B, m);
                        plasma_core_omp_ztrsm(
                            side, uplo, trans, diag,
                            mvbm, nvbk,
                            alpha, A(B.nt-k-1, B.nt-k-1), ldak,
                                   B(m,        B.nt-k-1), ldbm,
                            sequence, request);

                        for (int n = k+1; n < B.nt; n++) {
                            int ldan = plasma_tile_mmain(A, B.nt-1-n);
                            plasma_core_omp_zgemm(
                                PlasmaNoTrans, trans,
                                mvbm, B.nb, nvbk,
                                -1.0/alpha, B(m,        B.nt-k-1), ldbm,
                                            A(B.nt-1-n, B.nt-k-1), ldan,
                                1.0,        B(m,        B.nt-1-n), ldbm,
                                sequence, request);
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
                for (int k = 0; k < B.nt; k++) {
                    int nvbk = plasma_tile_nview(B, B.nt-k-1);
                    int ldak = plasma_tile_mmain(A, B.nt-k-1);
                    plasma_complex64_t lalpha = k == 0 ? alpha : 1.0;
                    for (int m = 0; m < B.mt; m++) {
                        int mvbm = plasma_tile_mview(B, m);
                        int ldbm = plasma_tile_mmain(B, m);
                        plasma_core_omp_ztrsm(
                            side, uplo, trans, diag,
                            mvbm, nvbk,
                            lalpha, A(B.nt-k-1, B.nt-k-1), ldak,
                                    B(m,        B.nt-k-1), ldbm,
                            sequence, request);

                        for (int n = k+1; n < B.nt; n++) {
                            plasma_core_omp_zgemm(
                                PlasmaNoTrans, PlasmaNoTrans,
                                mvbm, B.nb, nvbk,
                                -1.0,   B(m,        B.nt-k-1), ldbm,
                                        A(B.nt-1-k, B.nt-1-n), ldak,
                                lalpha, B(m,        B.nt-1-n), ldbm,
                                sequence, request);
                        }
                    }
                }
            }
            //==================================================
            //  PlasmaRight / PlasmaLower / Plasma[_Conj]Trans
            //==================================================
            else {
                for (int k = 0; k < B.nt; k++) {
                    int nvbk = plasma_tile_nview(B, k);
                    int ldak = plasma_tile_mmain(A, k);
                    for (int m = 0; m < B.mt; m++) {
                        int mvbm = plasma_tile_mview(B, m);
                        int ldbm = plasma_tile_mmain(B, m);
                        plasma_core_omp_ztrsm(
                            side, uplo, trans, diag,
                            mvbm, nvbk,
                            alpha, A(k, k), ldak,
                                   B(m, k), ldbm,
                            sequence, request);

                        for (int n = k+1; n < B.nt; n++) {
                            int nvbn = plasma_tile_nview(B, n);
                            int ldan = plasma_tile_mmain(A, n);
                            plasma_core_omp_zgemm(
                                PlasmaNoTrans, trans,
                                mvbm, nvbn, B.mb,
                                -1.0/alpha, B(m, k), ldbm,
                                            A(n, k), ldan,
                                1.0,        B(m, n), ldbm,
                                sequence, request);
                        }
                    }
                }
            }
        }
    }
}
