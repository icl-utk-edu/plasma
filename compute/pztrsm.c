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
#include "plasma_descriptor.h"
#include "plasma_types.h"
#include "plasma_internal.h"
#include "core_blas_z.h"

#define A(m, n) ((plasma_complex64_t*) plasma_tile_addr(A, m, n))
#define B(m, n) ((plasma_complex64_t*) plasma_tile_addr(B, m, n))
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

    plasma_complex64_t zone       = (plasma_complex64_t) 1.0;
    plasma_complex64_t mzone      = (plasma_complex64_t)-1.0;
    plasma_complex64_t minvalpha  = (plasma_complex64_t)-1.0 / alpha;
    plasma_complex64_t lalpha;

    // Check sequence status.
    if (sequence->status != PlasmaSuccess) {
        plasma_request_fail(sequence, request, PlasmaErrorSequence);
        return;
    }

    if (side == PlasmaLeft) {
        if (uplo == PlasmaUpper) {
            //============================================
            //  PlasmaLeft / PlasmaUpper / PlasmaNoTrans
            //============================================
            if (trans == PlasmaNoTrans) {
                for (k = 0; k < B.mt; k++) {
                    tempkm = k == 0 ? B.m-(B.mt-1)*B.mb : B.mb;
                    ldak = plasma_tile_mdim(A, B.mt-1-k);
                    ldbk = plasma_tile_mdim(B, B.mt-1-k);
                    lalpha = k == 0 ? alpha : zone;
                    for (n = 0; n < B.nt; n++) {
                        tempnn = n == B.nt-1 ? B.n-n*B.nb : B.nb;
                        core_omp_ztrsm(
                            side, uplo, trans, diag,
                            tempkm, tempnn,
                            lalpha, A(B.mt-1-k, B.mt-1-k), ldak,
                                    B(B.mt-1-k, n       ), ldbk);
                    }
                    for (m = k+1; m < B.mt; m++) {
                        ldam = plasma_tile_mdim(A, B.mt-1-m);
                        ldbm = plasma_tile_mdim(B, B.mt-1-m);
                        for (n = 0; n < B.nt; n++) {
                            tempnn = n == B.nt-1 ? B.n-n*B.nb : B.nb;
                            core_omp_zgemm(
                                PlasmaNoTrans, PlasmaNoTrans,
                                B.mb, tempnn, tempkm,
                                mzone,  A(B.mt-1-m, B.mt-1-k), ldam,
                                        B(B.mt-1-k, n       ), ldbk,
                                lalpha, B(B.mt-1-m, n       ), ldbm);
                        }
                    }
                }
            }
            //===============================================
            //  PlasmaLeft / PlasmaUpper / Plasma[Conj]Trans
            //===============================================
            else {
                for (k = 0; k < B.mt; k++) {
                    tempkm = k == B.mt-1 ? B.m-k*B.mb : B.mb;
                    ldak = plasma_tile_mdim(A, k);
                    ldbk = plasma_tile_mdim(B, k);
                    lalpha = k == 0 ? alpha : zone;
                    for (n = 0; n < B.nt; n++) {
                        tempnn = n == B.nt-1 ? B.n-n*B.nb : B.nb;
                        core_omp_ztrsm(
                            side, uplo, trans, diag,
                            tempkm, tempnn,
                            lalpha, A(k, k), ldak,
                                    B(k, n), ldbk);
                    }
                    for (m = k+1; m < B.mt; m++) {
                        tempmm = m == B.mt-1 ? B.m-m*B.mb : B.mb;
                        ldbm = plasma_tile_mdim(B, m);
                        for (n = 0; n < B.nt; n++) {
                            tempnn = n == B.nt-1 ? B.n-n*B.nb : B.nb;
                            core_omp_zgemm(
                                trans, PlasmaNoTrans,
                                tempmm, tempnn, B.mb,
                                mzone,  A(k, m), ldak,
                                        B(k, n), ldbk,
                                lalpha, B(m, n), ldbm);
                        }
                    }
                }
            }
        }
        else {
            //===========================================
            //  PlasmaLeft / PlasmaLower / PlasmaNoTrans
            //===========================================
            if (trans == PlasmaNoTrans) {
                for (k = 0; k < B.mt; k++) {
                    tempkm = k == B.mt-1 ? B.m-k*B.mb : B.mb;
                    ldak = plasma_tile_mdim(A, k);
                    ldbk = plasma_tile_mdim(B, k);
                    lalpha = k == 0 ? alpha : zone;
                    for (n = 0; n < B.nt; n++) {
                        tempnn = n == B.nt-1 ? B.n-n*B.nb : B.nb;
                        core_omp_ztrsm(
                            side, uplo, trans, diag,
                            tempkm, tempnn,
                            lalpha, A(k, k), ldak,
                                    B(k, n), ldbk);
                    }
                    for (m = k+1; m < B.mt; m++) {
                        tempmm = m == B.mt-1 ? B.m-m*B.mb : B.mb;
                        ldam = plasma_tile_mdim(A, m);
                        ldbm = plasma_tile_mdim(B, m);
                        for (n = 0; n < B.nt; n++) {
                            tempnn = n == B.nt-1 ? B.n-n*B.nb : B.nb;
                            core_omp_zgemm(
                                PlasmaNoTrans, PlasmaNoTrans,
                                tempmm, tempnn, B.mb,
                                mzone,  A(m, k), ldam,
                                        B(k, n), ldbk,
                                lalpha, B(m, n), ldbm);
                        }
                    }
                }
            }
            //===============================================
            //  PlasmaLeft / PlasmaLower / Plasma[Conj]Trans
            //===============================================
            else {
                for (k = 0; k < B.mt; k++) {
                    tempkm = k == 0 ? B.m-(B.mt-1)*B.mb : B.mb;
                    ldak = plasma_tile_mdim(A, B.mt-1-k);
                    ldbk = plasma_tile_mdim(B, B.mt-1-k);
                    lalpha = k == 0 ? alpha : zone;
                    for (n = 0; n < B.nt; n++) {
                        tempnn = n == B.nt-1 ? B.n-n*B.nb : B.nb;
                        core_omp_ztrsm(
                            side, uplo, trans, diag,
                            tempkm, tempnn,
                            lalpha, A(B.mt-1-k, B.mt-1-k), ldak,
                                    B(B.mt-1-k, n       ), ldbk);
                    }
                    for (m = k+1; m < B.mt; m++) {
                        tempmm = m == B.mt-1 ? B.m-m*B.mb : B.mb;
                        ldbm = plasma_tile_mdim(B, B.mt-1-m);
                        for (n = 0; n < B.nt; n++) {
                            tempnn = n == B.nt-1 ? B.n-n*B.nb : B.nb;
                            core_omp_zgemm(
                                trans, PlasmaNoTrans,
                                B.mb, tempnn, tempkm,
                                mzone,  A(B.mt-1-k, B.mt-1-m), ldak,
                                        B(B.mt-1-k, n       ), ldbk,
                                lalpha, B(B.mt-1-m, n       ), ldbm);
                        }
                    }
                }
            }
        }
    }
    else {
        if (uplo == PlasmaUpper) {
            //===========================================
            //  PlasmaRight / PlasmaUpper / PlasmaNoTrans
            //===========================================
            if (trans == PlasmaNoTrans) {
                for (k = 0; k < B.nt; k++) {
                    tempkn = k == B.nt-1 ? B.n-k*B.nb : B.nb;
                    ldak = plasma_tile_mdim(A, k);
                    lalpha = k == 0 ? alpha : zone;
                    for (m = 0; m < B.mt; m++) {
                        tempmm = m == B.mt-1 ? B.m-m*B.mb : B.mb;
                        ldbm = plasma_tile_mdim(B, m);
                        core_omp_ztrsm(
                            side, uplo, trans, diag,
                            tempmm, tempkn,
                            lalpha, A(k, k), ldak,
                                    B(m, k), ldbm);
                    }
                    for (m = 0; m < B.mt; m++) {
                        tempmm = m == B.mt-1 ? B.m-m*B.mb : B.mb;
                        ldbm = plasma_tile_mdim(B, m);
                        for (n = k+1; n < B.nt; n++) {
                            tempnn = n == B.nt-1 ? B.n-n*B.nb : B.nb;
                            core_omp_zgemm(
                                PlasmaNoTrans, PlasmaNoTrans,
                                tempmm, tempnn, B.mb,
                                mzone,  B(m, k), ldbm,
                                        A(k, n), ldak,
                                lalpha, B(m, n), ldbm);
                        }
                    }
                }
            }
            //===============================================
            //  PlasmaRight / PlasmaUpper / Plasma[Conj]Trans
            //===============================================
            else {
                for (k = 0; k < B.nt; k++) {
                    tempkn = k == 0 ? B.n-(B.nt-1)*B.nb : B.nb;
                    ldak = plasma_tile_mdim(A, B.nt-1-k);
                    for (m = 0; m < B.mt; m++) {
                        tempmm = m == B.mt-1 ? B.m-m*B.mb : B.mb;
                        ldbm = plasma_tile_mdim(B, m);
                        core_omp_ztrsm(
                            side, uplo, trans, diag,
                            tempmm, tempkn,
                            alpha, A(B.nt-1-k, B.nt-1-k), ldak,
                                   B(m,        B.nt-1-k), ldbm);

                        for (n = k+1; n < B.nt; n++) {
                            ldan = plasma_tile_mdim(A, B.nt-1-n);
                            core_omp_zgemm(
                                PlasmaNoTrans, trans,
                                tempmm, B.nb, tempkn,
                                minvalpha, B(m,        B.nt-1-k), ldbm,
                                           A(B.nt-1-n, B.nt-1-k), ldan,
                                zone,      B(m,        B.nt-1-n), ldbm);
                        }
                    }
                }
            }
        }
        else {
            //============================================
            //  PlasmaRight / PlasmaLower / PlasmaNoTrans
            //============================================
            if (trans == PlasmaNoTrans) {
                for (k = 0; k < B.nt; k++) {
                    tempkn = k == 0 ? B.n-(B.nt-1)*B.nb : B.nb;
                    ldak = plasma_tile_mdim(A, B.nt-1-k);
                    lalpha = k == 0 ? alpha : zone;
                    for (m = 0; m < B.mt; m++) {
                        tempmm = m == B.mt-1 ? B.m-m*B.mb : B.mb;
                        ldbm = plasma_tile_mdim(B, m);
                        core_omp_ztrsm(
                            side, uplo, trans, diag,
                            tempmm, tempkn,
                            lalpha, A(B.nt-1-k, B.nt-1-k), ldak,
                                    B(m,        B.nt-1-k), ldbm);

                        for (n = k+1; n < B.nt; n++) {
                            core_omp_zgemm(
                                PlasmaNoTrans, PlasmaNoTrans,
                                tempmm, B.nb, tempkn,
                                mzone,  B(m,        B.nt-1-k), ldbm,
                                        A(B.nt-1-k, B.nt-1-n), ldak,
                                lalpha, B(m,        B.nt-1-n), ldbm);
                        }
                    }
                }
            }
            //================================================
            //  PlasmaRight / PlasmaLower / Plasma[Conj]Trans
            //===============================================
            else {
                for (k = 0; k < B.nt; k++) {
                    tempkn = k == B.nt-1 ? B.n-k*B.nb : B.nb;
                    ldak = plasma_tile_mdim(A, k);
                    for (m = 0; m < B.mt; m++) {
                        tempmm = m == B.mt-1 ? B.m-m*B.mb : B.mb;
                        ldbm = plasma_tile_mdim(B, m);
                        core_omp_ztrsm(
                            side, uplo, trans, diag,
                            tempmm, tempkn,
                            alpha, A(k, k), ldak,
                                   B(m, k), ldbm);

                        for (n = k+1; n < B.nt; n++) {
                            tempnn = n == B.nt-1 ? B.n-n*B.nb : B.nb;
                            ldan = plasma_tile_mdim(A, n);
                            core_omp_zgemm(
                                PlasmaNoTrans, trans,
                                tempmm, tempnn, B.mb,
                                minvalpha, B(m, k), ldbm,
                                           A(n, k), ldan,
                                zone,      B(m, n), ldbm);
                        }
                    }
                }
            }
        }
    }
}
