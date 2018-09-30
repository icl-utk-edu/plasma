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

#define A(m,n) (plasma_complex64_t*)plasma_tile_addr(A, m, n)
#define B(m,n) (plasma_complex64_t*)plasma_tile_addr(B, m, n)

/***************************************************************************//**
 *  Parallel tile triangular solve - dynamic scheduling
 **/
void plasma_pztbsm(plasma_enum_t side, plasma_enum_t uplo,
                   plasma_enum_t trans, plasma_enum_t diag,
                   plasma_complex64_t alpha, plasma_desc_t A,
                                             plasma_desc_t B,
                   const int *ipiv,
                   plasma_sequence_t *sequence, plasma_request_t *request)
{
    // Return if failed sequence.
    if (sequence->status != PlasmaSuccess)
        return;

    if (side == PlasmaLeft) {
        if (uplo == PlasmaUpper) {
            if (trans == PlasmaNoTrans) {
                // ==========================================
                // PlasmaLeft / PlasmaUpper / PlasmaNoTrans
                // ==========================================
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
                                    B(B.mt-k-1,        n), ldbk,
                            sequence, request);
                    }
                    for (int m = imax(0, (B.mt-k-1)-A.kut+1); m < B.mt-k-1; m++) {
                        int ldam = plasma_tile_mmain(A, m);
                        int ldbm = plasma_tile_mmain(B, m);
                        for (int n = 0; n < B.nt; n++) {
                            int nvbn = plasma_tile_nview(B, n);
                            plasma_core_omp_zgemm(
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
            else {
                // ==============================================
                // PlasmaLeft / PlasmaUpper / Plasma[Conj]Trans
                // ==============================================
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
                    for (int m = k+1; m < imin(A.mt, k+A.kut); m++) {
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
            if (trans == PlasmaNoTrans) {
                // ==========================================
                // PlasmaLeft / PlasmaLower / PlasmaNoTrans
                // ==========================================
                for (int k = 0; k < B.mt; k++) {
                    int mvbk = plasma_tile_mview(B, k);
                    int ldak = plasma_tile_mmain(A, k);
                    int ldbk = plasma_tile_mmain(B, k);
                    plasma_complex64_t lalpha = k == 0 ? alpha : 1.0;
                    for (int n = 0; n < B.nt; n++) {
                        int nvbn = plasma_tile_nview(B, n);
                        if (ipiv != NULL) {
                            plasma_desc_t view = plasma_desc_view(B,
                                                                  0, n*A.nb,
                                                                  A.m, nvbn);
                            view.type = PlasmaGeneral;
                            // TODO: nested parallelization like getrf
                            #pragma omp taskwait
                            if (sequence->status == PlasmaSuccess) {
                                plasma_core_zgeswp(PlasmaRowwise, view, k*A.nb+1, k*A.nb+mvbk, ipiv, 1);
                            }
                        }
                        plasma_core_omp_ztrsm(
                            side, uplo, trans, diag,
                            mvbk, nvbn,
                            lalpha, A(k, k), ldak,
                                    B(k, n), ldbk,
                            sequence, request);
                    }
                    for (int m = k+1; m < imin(k+A.klt, A.mt); m++) {
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
            else {
                // ==============================================
                // PlasmaLeft / PlasmaLower / Plasma[Conj]Trans
                // ==============================================
                for (int k = 0; k < B.mt; k++) {
                    int mvbk = plasma_tile_mview(B, B.mt-k-1);
                    int ldak = plasma_tile_mmain(A, B.mt-k-1);
                    int ldbk = plasma_tile_mmain(B, B.mt-k-1);
                    plasma_complex64_t lalpha = k == 0 ? alpha : 1.0;
                    for (int m = (B.mt-k-1)+1; m < imin((B.mt-k-1)+A.klt, A.mt); m++) {
                        int mvbm = plasma_tile_mview(B, m);
                        int ldam = plasma_tile_mmain(A, m);
                        int ldbm = plasma_tile_mmain(B, m);
                        for (int n = 0; n < B.nt; n++) {
                            int nvbn = plasma_tile_nview(B, n);
                            plasma_core_omp_zgemm(
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
                        plasma_core_omp_ztrsm(
                            side, uplo, trans, diag,
                            mvbk, nvbn,
                            lalpha, A(B.mt-k-1, B.mt-k-1), ldak,
                                    B(B.mt-k-1,        n), ldbk,
                            sequence, request);
                        if (ipiv != NULL) {
                            int k1 = 1+(B.mt-k-1)*A.nb;
                            int k2 = k1+mvbk-1;
                            plasma_desc_t view = plasma_desc_view(B,
                                                                  0, n*A.nb,
                                                                  A.m, nvbn);
                            view.type = PlasmaGeneral;
                            #pragma omp taskwait
                            if (sequence->status == PlasmaSuccess) {
                                plasma_core_zgeswp(PlasmaRowwise, view, k1, k2, ipiv, -1);
                            }
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
