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
#include <plasma_core_blas.h>

#define A(m, n) (plasma_complex64_t*)plasma_tile_addr(A, m, n)
#define B(m, n) (plasma_complex64_t*)plasma_tile_addr(B, m, n)

/***************************************************************************//**
 *  Parallel tile triangular matrix-matrix multiplication.
 *  @see plasma_omp_ztrmm
 ******************************************************************************/
void plasma_pztrmm(plasma_enum_t side, plasma_enum_t uplo,
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
                for (int m = 0; m < B.mt; m++) {
                    int mvbm = plasma_tile_mview(B, m);
                    int ldbm = plasma_tile_mmain(B, m);
                    int ldam = plasma_tile_mmain(A, m);
                    for (int n = 0; n < B.nt; n++) {
                        int nvbn = plasma_tile_nview(B, n);
                        plasma_core_omp_ztrmm(
                            side, uplo, trans, diag,
                            mvbm, nvbn,
                            alpha, A(m, m), ldam,
                                   B(m, n), ldbm,
                            sequence, request);

                        for (int k = m+1; k < A.mt; k++) {
                            int nvak = plasma_tile_nview(A, k);
                            int ldbk = plasma_tile_mmain(B, k);
                            plasma_core_omp_zgemm(
                                trans, PlasmaNoTrans,
                                mvbm, nvbn, nvak,
                                alpha, A(m, k), ldam,
                                       B(k, n), ldbk,
                                1.0,   B(m, n), ldbm,
                                sequence, request);
                        }
                    }
                }
            }
            //================================================
            // PlasmaLeft / PlasmaUpper / Plasma[_Conj]Trans
            //================================================
            else {
                for (int m = B.mt-1; m > -1; m--) {
                    int mvbm = plasma_tile_mview(B, m);
                    int ldbm = plasma_tile_mmain(B, m);
                    int ldam = plasma_tile_mmain(A, m);
                    for (int n = 0; n < B.nt; n++) {
                        int nvbn = plasma_tile_nview(B, n);
                        plasma_core_omp_ztrmm(
                            side, uplo, trans, diag,
                            mvbm, nvbn,
                            alpha, A(m, m), ldam,
                                   B(m, n), ldbm,
                            sequence, request);

                        for (int k = 0; k < m; k++) {
                            int ldbk = plasma_tile_mmain(B, k);
                            int ldak = plasma_tile_mmain(A, k);
                            plasma_core_omp_zgemm(
                                trans, PlasmaNoTrans,
                                mvbm, nvbn, B.mb,
                                alpha, A(k, m), ldak,
                                       B(k, n), ldbk,
                                1.0,   B(m, n), ldbm,
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
                for (int m = B.mt-1; m > -1; m--) {
                    int mvbm = plasma_tile_mview(B, m);
                    int ldbm = plasma_tile_mmain(B, m);
                    int ldam = plasma_tile_mmain(A, m);
                    for (int n = 0; n < B.nt; n++) {
                        int nvbn = plasma_tile_nview(B, n);
                        plasma_core_omp_ztrmm(
                            side, uplo, trans, diag,
                            mvbm, nvbn,
                            alpha, A(m, m), ldam,
                                   B(m, n), ldbm,
                            sequence, request);

                        for (int k = 0; k < m; k++) {
                            int ldbk = plasma_tile_mmain(B, k);
                            plasma_core_omp_zgemm(
                                trans, PlasmaNoTrans,
                                mvbm, nvbn, B.mb,
                                alpha, A(m, k), ldam,
                                       B(k, n), ldbk,
                                1.0,   B(m, n), ldbm,
                                sequence, request);
                        }
                    }
                }
            }
            //================================================
            // PlasmaLeft / PlasmaLower / Plasma[_Conj]Trans
            //================================================
            else {
                for (int m = 0; m < B.mt; m++) {
                    int mvbm = plasma_tile_mview(B, m);
                    int ldbm = plasma_tile_mmain(B, m);
                    int ldam = plasma_tile_mmain(A, m);
                    for (int n = 0; n < B.nt; n++) {
                        int nvbn = plasma_tile_nview(B, n);
                        plasma_core_omp_ztrmm(
                            side, uplo, trans, diag,
                            mvbm, nvbn,
                            alpha, A(m, m), ldam,
                                   B(m, n), ldbm,
                            sequence, request);

                        for (int k = m+1; k < A.mt; k++) {
                            int mvak = plasma_tile_mview(A, k);
                            int ldak = plasma_tile_mmain(A, k);
                            int ldbk = plasma_tile_mmain(B, k);
                            plasma_core_omp_zgemm(
                                trans, PlasmaNoTrans,
                                mvbm, nvbn, mvak,
                                alpha, A(k, m), ldak,
                                       B(k, n), ldbk,
                                1.0,   B(m, n), ldbm,
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
                for (int n = B.nt-1; n > -1; n--) {
                    int nvbn = plasma_tile_nview(B, n);
                    int ldan = plasma_tile_mmain(A, n);
                    for (int m = 0; m < B.mt; m++) {
                        int mvbm = plasma_tile_mview(B, m);
                        int ldbm = plasma_tile_mmain(B, m);
                        plasma_core_omp_ztrmm(
                            side, uplo, trans, diag,
                            mvbm, nvbn,
                            alpha, A(n, n), ldan,
                                   B(m, n), ldbm,
                            sequence, request);

                        for (int k = 0; k < n; k++) {
                            int ldak = plasma_tile_mmain(A, k);
                            plasma_core_omp_zgemm(
                                PlasmaNoTrans, trans,
                                mvbm, nvbn, B.mb,
                                alpha, B(m, k), ldbm,
                                       A(k, n), ldak,
                                1.0,   B(m, n), ldbm,
                                sequence, request);
                        }
                    }
                }
            }
            //=================================================
            // PlasmaRight / PlasmaUpper / Plasma[_Conj]Trans
            //=================================================
            else {
                for (int n = 0; n < B.nt; n++) {
                    int nvbn = plasma_tile_nview(B, n);
                    int ldan = plasma_tile_mmain(A, n);
                    for (int m = 0; m < B.mt; m++) {
                        int mvbm = plasma_tile_mview(B, m);
                        int ldbm = plasma_tile_mmain(B, m);
                        plasma_core_omp_ztrmm(
                            side, uplo, trans, diag,
                            mvbm, nvbn,
                            alpha, A(n, n), ldan,
                                   B(m, n), ldbm,
                            sequence, request);

                        for (int k = n+1; k < A.mt; k++) {
                            int nvak = plasma_tile_nview(A, k);
                            plasma_core_omp_zgemm(
                                PlasmaNoTrans, trans,
                                mvbm, nvbn, nvak,
                                alpha, B(m, k), ldbm,
                                       A(n, k), ldan,
                                1.0,   B(m, n), ldbm,
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
                for (int n = 0; n < B.nt; n++) {
                    int nvbn = plasma_tile_nview(B, n);
                    int ldan = plasma_tile_mmain(A, n);
                    for (int m = 0; m < B.mt; m++) {
                        int mvbm = plasma_tile_mview(B, m);
                        int ldbm = plasma_tile_mmain(B, m);
                        plasma_core_omp_ztrmm(
                            side, uplo, trans, diag,
                            mvbm, nvbn,
                            alpha, A(n, n), ldan,
                                   B(m, n), ldbm,
                            sequence, request);

                        for (int k = n+1; k < A.mt; k++) {
                            int nvak = plasma_tile_nview(A, k);
                            int ldak = plasma_tile_mmain(A, k);
                            plasma_core_omp_zgemm(
                                PlasmaNoTrans, trans,
                                mvbm, nvbn, nvak,
                                alpha, B(m, k), ldbm,
                                       A(k, n), ldak,
                                1.0,   B(m, n), ldbm,
                                sequence, request);
                        }
                    }
                }
            }
            //=================================================
            // PlasmaRight / PlasmaLower / Plasma[_Conj]Trans
            //=================================================
            else {
                for (int n = B.nt-1; n > -1; n--) {
                    int nvbn = plasma_tile_nview(B, n);
                    int ldan = plasma_tile_mmain(A, n);
                    for (int m = 0; m < B.mt; m++) {
                        int mvbm = plasma_tile_mview(B, m);
                        int ldbm = plasma_tile_mmain(B, m);
                        plasma_core_omp_ztrmm(
                            side, uplo, trans, diag,
                            mvbm, nvbn,
                            alpha, A(n, n), ldan,
                                   B(m, n), ldbm,
                            sequence, request);

                        for (int k = 0; k < n; k++) {
                            plasma_core_omp_zgemm(
                                PlasmaNoTrans, trans,
                                mvbm, nvbn, B.mb,
                                alpha, B(m, k), ldbm,
                                       A(n, k), ldan,
                                1.0,   B(m, n), ldbm,
                                sequence, request);
                        }
                    }
                }
            }
        }
    }
}
