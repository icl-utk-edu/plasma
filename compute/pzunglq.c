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
#define Q(m, n) ((PLASMA_Complex64_t*) plasma_getaddr(Q, m, n))
#define T(m, n) ((PLASMA_Complex64_t*) plasma_getaddr(T, m, n))
/***************************************************************************//**
 *  Parallel construction of Q using tile V (application to identity)
 **/
void plasma_pzunglq(PLASMA_desc A, PLASMA_desc Q, PLASMA_desc T,
                    PLASMA_workspace *work,
                    PLASMA_sequence *sequence, PLASMA_request *request)
{
    int k, m, n;
    int ldak, ldqm;
    int tempnn, tempmm, tempkmin, tempkn;
    int tempAkm, tempAkn;
    int minmnt;

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

    minmnt = imin(A.mt, A.nt);
    for (k = minmnt-1; k >= 0; k--) {
        tempAkm  = k == A.mt-1 ? A.m-k*A.mb : A.mb;
        tempAkn  = k == A.nt-1 ? A.n-k*A.nb : A.nb;
        tempkmin = imin( tempAkn, tempAkm );
        tempkn   = k == Q.nt-1 ? Q.n-k*Q.nb : Q.nb;
        ldak = BLKLDD(A, k);
        for (n = Q.nt-1; n > k; n--) {
            tempnn = n == Q.nt-1 ? Q.n-n*Q.nb : Q.nb;
            for (m = k; m < Q.mt; m++) {
                tempmm = m == Q.mt-1 ? Q.m-m*Q.mb : Q.mb;
                ldqm = BLKLDD(Q, m);
                CORE_OMP_ztsmlq(
                    PlasmaRight, PlasmaNoTrans,
                    tempmm, Q.nb, tempmm, tempnn, tempAkm, ib, T.nb,
                    Q(m, k), ldqm,
                    Q(m, n), ldqm,
                    A(k, n), ldak,
                    T(k, n), T.mb,
                    work,
                    sequence, request);
            }
        }
        for (m = k; m < Q.mt; m++) {
            tempmm = m == Q.mt-1 ? Q.m-m*Q.mb : Q.mb;
            ldqm = BLKLDD(Q, m);
            CORE_OMP_zunmlq(
                PlasmaRight, PlasmaNoTrans,
                tempmm, tempkn, tempkmin, ib, T.nb,
                A(k, k), ldak,
                T(k, k), T.mb,
                Q(m, k), ldqm,
                work,
                sequence, request);
        }
    }
}
