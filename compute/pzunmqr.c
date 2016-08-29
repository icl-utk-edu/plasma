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
 *  Parallel application of Q using tile V - QR factorization
 * @see PLASMA_zgeqrs_Tile_Async
 **/
void plasma_pzunmqr(PLASMA_enum side, PLASMA_enum trans,
                    PLASMA_desc A, PLASMA_desc B, PLASMA_desc T,
                    PLASMA_sequence *sequence, PLASMA_request *request)
{
    int k, m, n;
    int ldak, ldbk, ldam, ldan, ldbm;
    int tempkm, tempnn, tempkmin, tempmm, tempkn;
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

    // PlasmaLeft / Plasma_ConjTrans
    if (side == PlasmaLeft ) {
        // Plasma_ConjTrans will be converted do PlasmaTrans in
        // automatic datatype conversion, which is what we  want here.
        // PlasmaConjTrans is protected from this conversion.
        if (trans == Plasma_ConjTrans) {
            for (k = 0; k < minMT; k++) {
                tempkm   = k == B.mt-1 ? B.m-k*B.mb : B.mb;
                tempkmin = k == minMT-1 ? minM-k*A.nb : A.nb;
                ldak = BLKLDD(A, k);
                ldbk = BLKLDD(B, k);
                for (n = 0; n < B.nt; n++) {
                    tempnn = n == B.nt-1 ? B.n-n*B.nb : B.nb;
                    CORE_OMP_zunmqr(
                        side, trans,
                        tempkm, tempnn, tempkmin, ib, T.nb,
                        A(k, k), ldak,
                        T(k, k), T.mb,
                        B(k, n), ldbk);
                }
                for (m = k+1; m < B.mt; m++) {
                    tempmm = m == B.mt-1 ? B.m-m*B.mb : B.mb;
                    ldam = BLKLDD(A, m);
                    ldbm = BLKLDD(B, m);
                    for (n = 0; n < B.nt; n++) {
                        tempnn = n == B.nt-1 ? B.n-n*B.nb : B.nb;
                        CORE_OMP_ztsmqr(
                            side, trans,
                            B.mb, tempnn, tempmm, tempnn, tempkmin, ib, T.nb,
                            B(k, n), ldbk,
                            B(m, n), ldbm,
                            A(m, k), ldam,
                            T(m, k), T.mb);
                    }
                }
            }
        }
        // PlasmaLeft / PlasmaNoTrans
        else {
            for (k = minMT-1; k >= 0; k--) {
                tempkm = k == B.mt-1 ? B.m-k*B.mb : B.mb;
                tempkmin = k == minMT-1 ? minM-k*A.nb : A.nb;
                ldak = BLKLDD(A, k);
                ldbk = BLKLDD(B, k);
                for (m = B.mt-1; m > k; m--) {
                    tempmm = m == B.mt-1 ? B.m-m*B.mb : B.mb;
                    ldam = BLKLDD(A, m);
                    ldbm = BLKLDD(B, m);
                    for (n = 0; n < B.nt; n++) {
                        tempnn = n == B.nt-1 ? B.n-n*B.nb : B.nb;
                        CORE_OMP_ztsmqr(
                            side, trans,
                            B.mb, tempnn, tempmm, tempnn, tempkmin, ib, T.nb,
                            B(k, n), ldbk,
                            B(m, n), ldbm,
                            A(m, k), ldam,
                            T(m, k), T.mb);
                    }
                }
                for (n = 0; n < B.nt; n++) {
                    tempnn = n == B.nt-1 ? B.n-n*B.nb : B.nb;
                    CORE_OMP_zunmqr(
                        side, trans,
                        tempkm, tempnn, tempkmin, ib, T.nb,
                        A(k, k), ldak,
                        T(k, k), T.mb,
                        B(k, n), ldbk);
                }
            }
        }
    }
    // PlasmaRight / Plasma_ConjTrans
    else {
        // Plasma_ConjTrans will be converted do PlasmaTrans in
        // automatic datatype conversion, which is what we want here.
        // PlasmaConjTrans is protected from this conversion.
        if (trans == Plasma_ConjTrans) {
            for (k = minMT-1; k >= 0; k--) {
                tempkn = k == B.nt-1 ? B.n-k*B.nb : B.nb;
                tempkmin = k == minMT-1 ? minM-k*A.nb : A.nb;
                ldak = BLKLDD(A, k);
                ldbk = BLKLDD(B, k);
                for (n = B.nt-1; n > k; n--) {
                    tempnn = n == B.nt-1 ? B.n-n*B.nb : B.nb;
                    ldan = BLKLDD(A, n);
                    for (m = 0; m < B.mt; m++) {
                        tempmm = m == B.mt-1 ? B.m-m*B.mb : B.mb;
                        ldbm = BLKLDD(B, m);
                        CORE_OMP_ztsmqr(
                            side, trans,
                            tempmm, B.nb, tempmm, tempnn, tempkmin, ib, T.nb,
                            B(m, k), ldbm,
                            B(m, n), ldbm,
                            A(n, k), ldan,
                            T(n, k), T.mb);
                    }
                }
                for (m = 0; m < B.mt; m++) {
                    tempmm = m == B.mt-1 ? B.m-m*B.mb : B.mb;
                    ldbm = BLKLDD(B, m);
                    CORE_OMP_zunmqr(
                        side, trans,
                        tempmm, tempkn, tempkmin, ib, T.nb,
                        A(k, k), ldak,
                        T(k, k), T.mb,
                        B(m, k), ldbm);
                }
            }
        }
        // PlasmaRight / PlasmaNoTrans
        else {
            for (k = 0; k < minMT; k++) {
                tempkn   = k == B.nt-1 ? B.n-k*B.nb : B.nb;
                tempkmin = k == minMT-1 ? minM-k*A.nb : A.nb;
                ldak = BLKLDD(A, k);
                for (m = 0; m < B.mt; m++) {
                    tempmm = m == B.mt-1 ? B.m-m*B.mb : B.mb;
                    ldbm = BLKLDD(B, m);
                    CORE_OMP_zunmqr(
                        side, trans,
                        tempmm, tempkn, tempkmin, ib, T.nb,
                        A(k, k), ldak,
                        T(k, k), T.mb,
                        B(m, k), ldbm);
                }
                for (n = k+1; n < B.nt; n++) {
                    tempnn = n == B.nt-1 ? B.n-n*B.nb : B.nb;
                    ldan = BLKLDD(A, n);
                    for (m = 0; m < B.mt; m++) {
                        tempmm = m == B.mt-1 ? B.m-m*B.mb : B.mb;
                        ldbm = BLKLDD(B, m);
                        CORE_OMP_ztsmqr(
                            side, trans,
                            tempmm, B.nb, tempmm, tempnn, tempkmin, ib, T.nb,
                            B(m, k), ldbm,
                            B(m, n), ldbm,
                            A(n, k), ldan,
                            T(n, k), T.mb);
                    }
                }
            }
        }
    }
}
