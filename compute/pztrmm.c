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
#include "plasma_descriptor.h"
#include "plasma_types.h"
#include "plasma_internal.h"
#include "core_blas_z.h"

#define A(m, n) ((PLASMA_Complex64_t*) plasma_getaddr(A, m, n))
#define B(m, n) ((PLASMA_Complex64_t*) plasma_getaddr(B, m, n))

/***************************************************************************//**
 *  Parallel tile triangular matrix-matrix multiplication.
 *  @see plasma_omp_ztrmm
 ******************************************************************************/
void plasma_pztrmm(PLASMA_enum side, PLASMA_enum uplo,
                   PLASMA_enum trans, PLASMA_enum diag,
                   PLASMA_Complex64_t alpha, PLASMA_desc A, PLASMA_desc B,
                   plasma_sequence_t *sequence, PLASMA_request *request)
{
    int k, m, n;
    int ldak, ldam, ldan, ldbk, ldbm;
    int tempkm, tempkn, tempmm, tempnn;

    PLASMA_Complex64_t zone = (PLASMA_Complex64_t)1.0;

    // Check sequence status
    if (sequence->status != PLASMA_SUCCESS) {
        plasma_request_fail(sequence, request, PLASMA_ERR_SEQUENCE_FLUSHED);
        return;
    }

    if (side == PlasmaLeft) {
        if (uplo == PlasmaUpper) {
            //==========================================
            // PlasmaLeft / PlasmaUpper / PlasmaNoTrans
            //==========================================
            if (trans == PlasmaNoTrans) {
                for (m = 0; m < B.mt; m++) {
                    tempmm = m == B.mt-1 ? B.m-m*B.mb : B.mb;
                    ldbm = BLKLDD(B, m);
                    ldam = BLKLDD(A, m);
                    for (n = 0; n < B.nt; n++) {
                        tempnn = n == B.nt-1 ? B.n-n*B.nb : B.nb;
                        core_omp_ztrmm(
                            side, uplo, trans, diag,
                            tempmm, tempnn,
                            alpha, A(m, m), ldam,
                                   B(m, n), ldbm);

                        for (k = m+1; k < A.mt; k++) {
                            tempkn = k == A.nt-1 ? A.n-k*A.nb : A.nb;
                            ldbk = BLKLDD(B, k);
                            core_omp_zgemm(
                                trans, PlasmaNoTrans,
                                tempmm, tempnn, tempkn,
                                alpha, A(m, k), ldam,
                                       B(k, n), ldbk,
                                zone,  B(m, n), ldbm);
                        }
                    }
                }
            }
            //==============================================
            // PlasmaLeft / PlasmaUpper / Plasma[Conj]Trans
            //==============================================
            else {
                for (m = B.mt-1; m > -1; m--) {
                    tempmm = m == B.mt-1 ? B.m-m*B.mb : B.mb;
                    ldbm = BLKLDD(B, m);
                    ldam = BLKLDD(A, m);
                    for (n = 0; n < B.nt; n++) {
                        tempnn = n == B.nt-1 ? B.n-n*B.nb : B.nb;
                        core_omp_ztrmm(
                            side, uplo, trans, diag,
                            tempmm, tempnn,
                            alpha, A(m, m), ldam,
                                   B(m, n), ldbm);

                        for (k = 0; k < m; k++) {
                            ldbk = BLKLDD(B, k);
                            ldak = BLKLDD(A, k);
                            core_omp_zgemm(
                                trans, PlasmaNoTrans,
                                tempmm, tempnn, B.mb,
                                alpha, A(k, m), ldak,
                                       B(k, n), ldbk,
                                zone,  B(m, n), ldbm);
                        }
                    }
                }
            }
        }
        else {
            //==========================================
            // PlasmaLeft / PlasmaLower / PlasmaNoTrans
            //==========================================
            if (trans == PlasmaNoTrans) {
                for (m = B.mt-1; m > -1; m--) {
                    tempmm = m == B.mt-1 ? B.m-m*B.mb : B.mb;
                    ldbm = BLKLDD(B, m);
                    ldam = BLKLDD(A, m);
                    for (n = 0; n < B.nt; n++) {
                        tempnn = n == B.nt-1 ? B.n-n*B.nb : B.nb;
                        core_omp_ztrmm(
                            side, uplo, trans, diag,
                            tempmm, tempnn,
                            alpha, A(m, m), ldam,
                                   B(m, n), ldbm);

                        for (k = 0; k < m; k++) {
                            ldbk = BLKLDD(B, k);
                            core_omp_zgemm(
                                trans, PlasmaNoTrans,
                                tempmm, tempnn, B.mb,
                                alpha, A(m, k), ldam,
                                       B(k, n), ldbk,
                                zone,  B(m, n), ldbm);
                        }
                    }
                }
            }
            //==============================================
            // PlasmaLeft / PlasmaLower / Plasma[Conj]Trans
            //==============================================
            else {
                for (m = 0; m < B.mt; m++) {
                    tempmm = m == B.mt-1 ? B.m-m*B.mb : B.mb;
                    ldbm = BLKLDD(B, m);
                    ldam = BLKLDD(A, m);
                    for (n = 0; n < B.nt; n++) {
                        tempnn = n == B.nt-1 ? B.n-n*B.nb : B.nb;
                        core_omp_ztrmm(
                            side, uplo, trans, diag,
                            tempmm, tempnn,
                            alpha, A(m, m), ldam,
                                   B(m, n), ldbm);

                        for (k = m+1; k < A.mt; k++) {
                            tempkm = k == A.mt-1 ? A.m-k*A.mb : A.mb;
                            ldak = BLKLDD(A, k);
                            ldbk = BLKLDD(B, k);
                            core_omp_zgemm(
                                trans, PlasmaNoTrans,
                                tempmm, tempnn, tempkm,
                                alpha, A(k, m), ldak,
                                       B(k, n), ldbk,
                                zone,  B(m, n), ldbm);
                        }
                    }
                }
            }
        }
    }
    else {
        if (uplo == PlasmaUpper) {
            //===========================================
            // PlasmaRight / PlasmaUpper / PlasmaNoTrans
            //===========================================
            if (trans == PlasmaNoTrans) {
                for (n = B.nt-1; n > -1; n--) {
                    tempnn = n == B.nt-1 ? B.n-n*B.nb : B.nb;
                    ldan = BLKLDD(A, n);
                    for (m = 0; m < B.mt; m++) {
                        tempmm = m == B.mt-1 ? B.m-m*B.mb : B.mb;
                        ldbm = BLKLDD(B, m);
                        core_omp_ztrmm(
                            side, uplo, trans, diag,
                            tempmm, tempnn,
                            alpha, A(n, n), ldan,
                                   B(m, n), ldbm);

                        for (k = 0; k < n; k++) {
                            ldak = BLKLDD(A, k);
                            core_omp_zgemm(
                                PlasmaNoTrans, trans,
                                tempmm, tempnn, B.mb,
                                alpha, B(m, k), ldbm,
                                       A(k, n), ldak,
                                zone,  B(m, n), ldbm);
                        }
                    }
                }
            }
            //===============================================
            // PlasmaRight / PlasmaUpper / Plasma[Conj]Trans
            //===============================================
            else {
                for (n = 0; n < B.nt; n++) {
                    tempnn = n == B.nt-1 ? B.n-n*B.nb : B.nb;
                    ldan = BLKLDD(A, n);
                    for (m = 0; m < B.mt; m++) {
                        tempmm = m == B.mt-1 ? B.m-m*B.mb : B.mb;
                        ldbm = BLKLDD(B, m);
                        core_omp_ztrmm(
                            side, uplo, trans, diag,
                            tempmm, tempnn,
                            alpha, A(n, n), ldan,
                                   B(m, n), ldbm);

                        for (k = n+1; k < A.mt; k++) {
                            tempkn = k == A.nt-1 ? A.n-k*A.nb : A.nb;
                            core_omp_zgemm(
                                PlasmaNoTrans, trans,
                                tempmm, tempnn, tempkn,
                                alpha, B(m, k), ldbm,
                                       A(n, k), ldan,
                                zone,  B(m, n), ldbm);
                        }
                    }
                }
            }
        }
        else {
            //===========================================
            // PlasmaRight / PlasmaLower / PlasmaNoTrans
            //===========================================
            if (trans == PlasmaNoTrans) {
                for (n = 0; n < B.nt; n++) {
                    tempnn = n == B.nt-1 ? B.n-n*B.nb : B.nb;
                    ldan = BLKLDD(A, n);
                    for (m = 0; m < B.mt; m++) {
                        tempmm = m == B.mt-1 ? B.m-m*B.mb : B.mb;
                        ldbm = BLKLDD(B, m);
                        core_omp_ztrmm(
                            side, uplo, trans, diag,
                            tempmm, tempnn,
                            alpha, A(n, n), ldan,
                                   B(m, n), ldbm);

                        for (k = n+1; k < A.mt; k++) {
                            tempkn = k == A.nt-1 ? A.n-k*A.nb : A.nb;
                            ldak = BLKLDD(A, k);
                            core_omp_zgemm(
                                PlasmaNoTrans, trans,
                                tempmm, tempnn, tempkn,
                                alpha, B(m, k), ldbm,
                                       A(k, n), ldak,
                                zone,  B(m, n), ldbm);
                        }
                    }
                }
            }
            //===============================================
            // PlasmaRight / PlasmaLower / Plasma[Conj]Trans
            //===============================================
            else {
                for (n = B.nt-1; n > -1; n--) {
                    tempnn = n == B.nt-1 ? B.n-n*B.nb : B.nb;
                    ldan = BLKLDD(A, n);
                    for (m = 0; m < B.mt; m++) {
                        tempmm = m == B.mt-1 ? B.m-m*B.mb : B.mb;
                        ldbm = BLKLDD(B, m);
                        core_omp_ztrmm(
                            side, uplo, trans, diag,
                            tempmm, tempnn,
                            alpha, A(n, n), ldan,
                                   B(m, n), ldbm);

                        for (k = 0; k < n; k++) {
                            core_omp_zgemm(
                                PlasmaNoTrans, trans,
                                tempmm, tempnn, B.mb,
                                alpha, B(m, k), ldbm,
                                       A(n, k), ldan,
                                zone,  B(m, n), ldbm);
                        }
                    }
                }
            }
        }
    }
}
