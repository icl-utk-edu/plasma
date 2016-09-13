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

#define A(m, n) ((PLASMA_Complex64_t*) plasma_getaddr(A, m, n))
#define B(m, n) ((PLASMA_Complex64_t*) plasma_getaddr(B, m, n))
/***************************************************************************//**
 * Parallel tile triangular solve.
 * @see plasma_omp_ztrsm
 ******************************************************************************/
void plasma_pztrsm(PLASMA_enum side, PLASMA_enum uplo,
                   PLASMA_enum trans, PLASMA_enum diag,
                   PLASMA_Complex64_t alpha, PLASMA_desc A,
                                             PLASMA_desc B,
                   plasma_sequence_t *sequence, plasma_request_t *request)
{
    int k, m, n;
    int ldak, ldam, ldan, ldbk, ldbm;
    int tempkm, tempkn, tempmm, tempnn;

    PLASMA_Complex64_t zone       = (PLASMA_Complex64_t) 1.0;
    PLASMA_Complex64_t mzone      = (PLASMA_Complex64_t)-1.0;
    PLASMA_Complex64_t minvalpha  = (PLASMA_Complex64_t)-1.0 / alpha;
    PLASMA_Complex64_t lalpha;

    // Check sequence status.
    if (sequence->status != PLASMA_SUCCESS) {
        plasma_request_fail(sequence, request, PLASMA_ERR_SEQUENCE_FLUSHED);
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
                    ldak = BLKLDD(A, B.mt-1-k);
                    ldbk = BLKLDD(B, B.mt-1-k);
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
                        ldam = BLKLDD(A, B.mt-1-m);
                        ldbm = BLKLDD(B, B.mt-1-m);
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
                    ldak = BLKLDD(A, k);
                    ldbk = BLKLDD(B, k);
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
                        ldbm = BLKLDD(B, m);
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
                    ldak = BLKLDD(A, k);
                    ldbk = BLKLDD(B, k);
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
                        ldam = BLKLDD(A, m);
                        ldbm = BLKLDD(B, m);
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
                    ldak = BLKLDD(A, B.mt-1-k);
                    ldbk = BLKLDD(B, B.mt-1-k);
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
                        ldbm = BLKLDD(B, B.mt-1-m);
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
                    ldak = BLKLDD(A, k);
                    lalpha = k == 0 ? alpha : zone;
                    for (m = 0; m < B.mt; m++) {
                        tempmm = m == B.mt-1 ? B.m-m*B.mb : B.mb;
                        ldbm = BLKLDD(B, m);
                        core_omp_ztrsm(
                            side, uplo, trans, diag,
                            tempmm, tempkn,
                            lalpha, A(k, k), ldak,
                                    B(m, k), ldbm);
                    }
                    for (m = 0; m < B.mt; m++) {
                        tempmm = m == B.mt-1 ? B.m-m*B.mb : B.mb;
                        ldbm = BLKLDD(B, m);
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
                    ldak = BLKLDD(A, B.nt-1-k);
                    for (m = 0; m < B.mt; m++) {
                        tempmm = m == B.mt-1 ? B.m-m*B.mb : B.mb;
                        ldbm = BLKLDD(B, m);
                        core_omp_ztrsm(
                            side, uplo, trans, diag,
                            tempmm, tempkn,
                            alpha, A(B.nt-1-k, B.nt-1-k), ldak,
                                   B(m,        B.nt-1-k), ldbm);

                        for (n = k+1; n < B.nt; n++) {
                            ldan = BLKLDD(A, B.nt-1-n);
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
                    ldak = BLKLDD(A, B.nt-1-k);
                    lalpha = k == 0 ? alpha : zone;
                    for (m = 0; m < B.mt; m++) {
                        tempmm = m == B.mt-1 ? B.m-m*B.mb : B.mb;
                        ldbm = BLKLDD(B, m);
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
                    ldak = BLKLDD(A, k);
                    for (m = 0; m < B.mt; m++) {
                        tempmm = m == B.mt-1 ? B.m-m*B.mb : B.mb;
                        ldbm = BLKLDD(B, m);
                        core_omp_ztrsm(
                            side, uplo, trans, diag,
                            tempmm, tempkn,
                            alpha, A(k, k), ldak,
                                   B(m, k), ldbm);

                        for (n = k+1; n < B.nt; n++) {
                            tempnn = n == B.nt-1 ? B.n-n*B.nb : B.nb;
                            ldan = BLKLDD(A, n);
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
