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
#include "plasma_types.h"
#include "plasma_internal.h"
#include "core_blas_z.h"

#define A(m, n) ((PLASMA_Complex64_t*) plasma_getaddr(A, m, n))
#define B(m, n) ((PLASMA_Complex64_t*) plasma_getaddr(B, m, n))
#define T(m, n) ((PLASMA_Complex64_t*) plasma_getaddr(T, m, n))
/***************************************************************************//**
 *  Parallel application of Q using tile V - LQ factorization
 * @see plasma_omp_zgelqs
 **/
void plasma_pzunmlq(PLASMA_enum side, PLASMA_enum trans,
                    PLASMA_desc A, PLASMA_desc B, PLASMA_desc T,
                    PLASMA_workspace *work,
                    plasma_sequence_t *sequence, plasma_request_t *request)
{
    int k, m, n;
    int ldak, ldbk, ldbm;
    int tempmm, tempnn, tempkn, tempkm, tempkmin;
    int minMT, minM;

    if (sequence->status != PLASMA_SUCCESS)
        return;

    // Set inner blocking from the plasma context
    plasma_context_t *plasma = plasma_context_self();
    if (plasma == NULL) {
        plasma_error("PLASMA not initialized");
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        return;
    }
    int ib = plasma->ib;

    if (A.m > A.n) {
        minM  = A.n;
        minMT = A.nt;
    }
    else {
        minM  = A.m;
        minMT = A.mt;
    }

    if (side == PlasmaLeft ) {
        if (trans == PlasmaNoTrans) {
            // PlasmaLeft / PlasmaNoTrans
            for (k = 0; k < minMT; k++) {
                tempkm   = k == B.mt -1 ? B.m -k*B.mb : B.mb;
                tempkmin = k == minMT-1 ? minM-k*A.nb : A.nb;
                ldak = BLKLDD(A, k);
                ldbk = BLKLDD(B, k);
                for (n = 0; n < B.nt; n++) {
                    tempnn = n == B.nt-1 ? B.n-n*B.nb : B.nb;
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
                    tempmm = m == B.mt-1 ? B.m-m*B.mb : B.mb;
                    ldbm = BLKLDD(B, m);
                    for (n = 0; n < B.nt; n++) {
                        tempnn = n == B.nt-1 ? B.n-n*B.nb : B.nb;
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
        else {
            // PlasmaLeft / Plasma_ConjTrans
            for (k = minMT-1; k >= 0; k--) {
                tempkm   = k == B.mt -1 ? B.m -k*B.mb : B.mb;
                tempkmin = k == minMT-1 ? minM-k*A.nb : A.nb;
                ldak = BLKLDD(A, k);
                ldbk = BLKLDD(B, k);
                for (m = B.mt-1; m > k; m--) {
                    tempmm = m == B.mt-1 ? B.m-m*B.mb : B.mb;
                    ldbm = BLKLDD(B, m);
                    for (n = 0; n < B.nt; n++) {
                        tempnn   = n == B.nt-1 ? B.n-n*B.nb : B.nb;
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
                    tempnn = n == B.nt-1 ? B.n-n*B.nb : B.nb;
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
        if (trans == PlasmaNoTrans) {
            // PlasmaRight / PlasmaNoTrans
            for (k = minMT-1; k >= 0; k--) {
                tempkn   = k == B.nt -1 ? B.n -k*B.nb : B.nb;
                tempkmin = k == minMT-1 ? minM-k*A.nb : A.nb;
                ldak = BLKLDD(A, k);
                for (n = B.nt-1; n > k; n--) {
                    tempnn = n == B.nt-1 ? B.n-n*B.nb : B.nb;
                    for (m = 0; m < B.mt; m++) {
                        tempmm = m == B.mt-1 ? B.m-m*B.mb : B.mb;
                        ldbm = BLKLDD(B, m);
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
                    tempmm = m == B.mt-1 ? B.m-m*B.mb : B.mb;
                    ldbm = BLKLDD(B, m);
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
        else {
            // PlasmaRight / Plasma_ConjTrans
            for (k = 0; k < minMT; k++) {
                tempkn   = k == B.nt -1 ? B.n -k*B.nb : B.nb;
                tempkmin = k == minMT-1 ? minM-k*A.mb : A.mb;
                ldak = BLKLDD(A, k);
                for (m = 0; m < B.mt; m++) {
                    tempmm = m == B.mt-1 ? B.m-m*B.mb : B.mb;
                    ldbm = BLKLDD(B, m);
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
                    tempnn = n == B.nt-1 ? B.n-n*B.nb : B.nb;
                    for (m = 0; m < B.mt; m++) {
                        tempmm = m == B.mt-1 ? B.m-m*B.mb : B.mb;
                        ldbm = BLKLDD(B, m);
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
