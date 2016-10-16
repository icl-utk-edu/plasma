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

#define A(m,n) (plasma_complex64_t*)plasma_tile_addr(A, m, n)
#define B(m,n) (plasma_complex64_t*)plasma_tile_addr(B, m, n)
#define IPIV(k) &(IPIV[B.mb*(k)])

/***************************************************************************//**
 *  Parallel tile triangular solve - dynamic scheduling
 **/
void plasma_pztbsm(plasma_enum_t side, plasma_enum_t uplo,
                   plasma_enum_t trans, plasma_enum_t diag,
                   plasma_complex64_t alpha, plasma_desc_t A,
                                             plasma_desc_t B,
                   const int *IPIV,
                   plasma_sequence_t *sequence, plasma_request_t *request)
{
    // Check sequence status.
    if (sequence->status != PlasmaSuccess) {
        plasma_request_fail(sequence, request, PlasmaErrorSequence);
        return;
    }

    if (side == PlasmaLeft) {
        if (uplo == PlasmaUpper) {
            // ==========================================
            // PlasmaLeft / PlasmaUpper / PlasmaNoTrans
            // ==========================================
            if (trans == PlasmaNoTrans) {
                for (int k = 0; k < B.mt; k++) {
                    int mvbk = plasma_tile_mview(B, B.mt-k-1);
                    int ldak = BLKLDD_BAND(uplo, A, B.mt-k-1, B.mt-k-1);
                    int ldbk = plasma_tile_mmain(B, B.mt-k-1);
                    plasma_complex64_t lalpha = k == 0 ? alpha : 1.0;
                    for (int n = 0; n < B.nt; n++) {
                        int nvbn = plasma_tile_nview(B, n);
                        core_omp_ztrsm(
                            side, uplo, trans, diag,
                            mvbk, nvbn,
                            lalpha, A(B.mt-k-1, B.mt-k-1), ldak,
                                    B(B.mt-k-1,        n), ldbk,
                            sequence, request);
                    }
                    for (int m = imax(0, (B.mt-k-1)-A.kut+1); m < B.mt-k-1; m++) {
                        int ldam = BLKLDD_BAND(uplo, A, m, B.mt-k-1);
                        int ldbm = plasma_tile_mmain(B, m);
                        for (int n = 0; n < B.nt; n++) {
                            int nvbn = plasma_tile_nview(B, n);
                            core_omp_zgemm(
                                PlasmaNoTrans, PlasmaNoTrans,
                                B.mb, nvbn, mvbk,
                                -1.0,   A(m, B.mt-k-1), ldam,
                                        B(B.mt-k-1, n), ldbk,
                                lalpha, B(m, n       ), ldbm,
                                sequence, request);
                        }
                    }
                }
            }
            // ==============================================
            // PlasmaLeft / PlasmaUpper / Plasma[Conj]Trans
            // ==============================================
            else {
                for (int k = 0; k < B.mt; k++) {
                    int mvbk = plasma_tile_mview(B, k);
                    int ldak = BLKLDD_BAND(uplo, A, k, k);
                    int ldbk = plasma_tile_mmain(B, k);
                    plasma_complex64_t lalpha = k == 0 ? alpha : 1.0;
                    for (int n = 0; n < B.nt; n++) {
                        int nvbn = plasma_tile_nview(B, n);
                        core_omp_ztrsm(
                            side, uplo, trans, diag,
                            mvbk, nvbn,
                            lalpha, A(k, k), ldak,
                                    B(k, n), ldbk,
                            sequence, request);
                    }
                    for (int m = k+1; m < imin(A.mt, k+A.kut); m++) {
                        int mvbm = plasma_tile_mview(B, m);
                        int ldam = BLKLDD_BAND(uplo, A, k, m);
                        int ldbm = plasma_tile_mmain(B, m);
                        for (int n = 0; n < B.nt; n++) {
                            int nvbn = plasma_tile_nview(B, n);
                            core_omp_zgemm(
                                trans, PlasmaNoTrans,
                                mvbm, nvbn, B.mb,
                                -1.0,   A(k, m), ldam,
                                        B(k, n), ldbk,
                                lalpha, B(m, n), ldbm,
                                sequence, request);
                        }
                    }
                }
            }
        }
        else {
            // ==========================================
            // PlasmaLeft / PlasmaLower / PlasmaNoTrans
            // ==========================================
            if (trans == PlasmaNoTrans) {
                for (int k = 0; k < B.mt; k++) {
                    int mvbk = plasma_tile_mview(B, k);
                    int ldak = BLKLDD_BAND(uplo, A, B.mt-k-1, B.mt-k-1);
                    int ldbk = plasma_tile_mmain(B, k);
                    plasma_complex64_t lalpha = k == 0 ? alpha : 1.0;
                    for (int n = 0; n < B.nt; n++) {
                        int nvbn = plasma_tile_nview(B, n);
                        if (IPIV != NULL) {
                            #ifdef ZLASWP_ONTILE
                            // commented out because it takes descriptor
                            tempi = k*B.mb;
                            core_omp_zlaswp_ontile(
                                B, k, n, B.m-tempi, tempnn,
                                1, tempkm, IPIV(k), 1);
                            #endif
                        }
                        core_omp_ztrsm(
                            side, uplo, trans, diag,
                            mvbk, nvbn,
                            lalpha, A(k, k), ldak,
                                    B(k, n), ldbk,
                            sequence, request);
                    }
                    for (int m = k+1; m < imin(k+A.klt, A.mt); m++) {
                        int mvbm = plasma_tile_mview(B, m);
                        int ldam = BLKLDD_BAND(uplo, A, m, k);
                        int ldbm = plasma_tile_mmain(B, m);
                        for (int n = 0; n < B.nt; n++) {
                            int nvbn = plasma_tile_nview(B, n);
                            core_omp_zgemm(
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
            // ==============================================
            // PlasmaLeft / PlasmaLower / Plasma[Conj]Trans
            // ==============================================
            else {
                for (int k = 0; k < B.mt; k++) {
                    int mvbk = plasma_tile_mview(B, B.mt-k-1);
                    int ldak = BLKLDD_BAND(uplo, A, B.mt-k-1, B.mt-k-1);
                    int ldbk = plasma_tile_mmain(B, B.mt-k-1);
                    plasma_complex64_t lalpha = k == 0 ? alpha : 1.0;
                    for (int m = (B.mt-k-1)+1; m < imin((B.mt-k-1)+A.klt, A.mt); m++) {
                        int mvbm = plasma_tile_mview(B, m);
                        int ldam = BLKLDD_BAND(uplo, A, m, B.mt-k-1);
                        int ldbm = plasma_tile_mmain(B, m);
                        for (int n = 0; n < B.nt; n++) {
                            int nvbn = plasma_tile_nview(B, n);
                            core_omp_zgemm(
                                trans, PlasmaNoTrans,
                                mvbk, nvbn, mvbm,
                                -1.0,   A(m, B.mt-k-1), ldam,
                                        B(m, n       ), ldbm,
                                lalpha, B(B.mt-k-1, n), ldbk,
                                sequence, request);
                        }
                    }
                    for (int n = 0; n < B.nt; n++) {
                        int nvbn = plasma_tile_nview(B, n);
                        core_omp_ztrsm(
                            side, uplo, trans, diag,
                            mvbk, nvbn,
                            lalpha, A(B.mt-k-1, B.mt-k-1), ldak,
                                    B(B.mt-k-1,        n), ldbk,
                            sequence, request);
                        if (IPIV != NULL) {
                            #ifdef ZLASWP_ONTILE
                            // commented out because it takes descriptor
                            int tempi = (B.mt-k-1)*B.mb;
                            core_omp_zlaswp_ontile(
                                B, B.mt-k-1, n, B.m-tempi, tempnn,
                                1, tempkm, IPIV(B.mt-k-1), -1);
                            #endif
                        }
                    }
                }
            }
        }
    }
    else {
        /*
         *  TODO: triangular-solve from right.
         */
    }
    return;
}
