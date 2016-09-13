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
#define T(m, n) ((PLASMA_Complex64_t*) plasma_getaddr(T, m, n))
/***************************************************************************//**
 *  Parallel tile LQ factorization - dynamic scheduling
 * @see plasma_omp_zgelqf
 **/
void plasma_pzgelqf(plasma_desc_t A, plasma_desc_t T,
                    PLASMA_workspace *work,
                    plasma_sequence_t *sequence, plasma_request_t *request)
{
    int k, m, n;
    int ldak, ldam;
    int tempkm, tempkn, tempmm, tempnn;

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

    for (k = 0; k < imin(A.mt, A.nt); k++) {
        tempkm = k == A.mt-1 ? A.m-k*A.mb : A.mb;
        tempkn = k == A.nt-1 ? A.n-k*A.nb : A.nb;
        ldak = BLKLDD(A, k);
        core_omp_zgelqt(
            tempkm, tempkn, ib, T.nb,
            A(k, k), ldak,
            T(k, k), T.mb,
            work,
            sequence, request);

        for (m = k+1; m < A.mt; m++) {
            tempmm = m == A.mt-1 ? A.m-m*A.mb : A.mb;
            ldam = BLKLDD(A, m);
            // Plasma_ConjTrans will be converted to PlasmaTrans in
            // automatic datatype conversion, which is what we
            // want here.
            // PlasmaConjTrans is protected from this conversion.
            core_omp_zunmlq(
                PlasmaRight, Plasma_ConjTrans,
                tempmm, tempkn, tempkn, ib, T.nb,
                A(k, k), ldak,
                T(k, k), T.mb,
                A(m, k), ldam,
                work,
                sequence, request);
        }
        for (n = k+1; n < A.nt; n++) {
            tempnn = n == A.nt-1 ? A.n-n*A.nb : A.nb;
            core_omp_ztslqt(
                tempkm, tempnn, ib, T.nb,
                A(k, k), ldak,
                A(k, n), ldak,
                T(k, n), T.mb,
                work,
                sequence, request);

            for (m = k+1; m < A.mt; m++) {
                tempmm = m == A.mt-1 ? A.m-m*A.mb : A.mb;
                ldam = BLKLDD(A, m);
                core_omp_ztsmlq(
                    PlasmaRight, Plasma_ConjTrans,
                    tempmm, A.nb, tempmm, tempnn, A.mb, ib, T.nb,
                    A(m, k), ldam,
                    A(m, n), ldam,
                    A(k, n), ldak,
                    T(k, n), T.mb,
                    work,
                    sequence, request);
            }
        }
    }
}
