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

#define A(m, n) ((plasma_complex64_t*) plasma_tile_addr(A, m, n))
#define T(m, n) ((plasma_complex64_t*) plasma_tile_addr(T, m, n))
/***************************************************************************//**
 *  Parallel tile QR factorization - dynamic scheduling
 * @see plasma_omp_zgeqrf
 **/
void plasma_pzgeqrf(plasma_desc_t A, plasma_desc_t T,
                    plasma_workspace_t *work,
                    plasma_sequence_t *sequence, plasma_request_t *request)
{
    int k, m, n;
    int ldak, ldam;
    int tempkm, tempkn, tempnn, tempmm;

    // Check sequence status.
    if (sequence->status != PlasmaSuccess) {
        plasma_request_fail(sequence, request, PlasmaErrorSequence);
        return;
    }

    // Set inner blocking from the plasma context
    plasma_context_t *plasma = plasma_context_self();
    if (plasma == NULL) {
        plasma_error("PLASMA not initialized");
        plasma_request_fail(sequence, request, PlasmaErrorIllegalValue);
        return;
    }
    int ib = plasma->ib;

    for (k = 0; k < imin(A.mt, A.nt); k++) {
        tempkm = k == A.mt-1 ? A.m-k*A.mb : A.mb;
        tempkn = k == A.nt-1 ? A.n-k*A.nb : A.nb;
        ldak = BLKLDD(A, k);
        core_omp_zgeqrt(
            tempkm, tempkn, ib, T.nb,
            A(k, k), ldak,
            T(k, k), T.mb,
            work,
            sequence, request);

        for (n = k+1; n < A.nt; n++) {
            tempnn = n == A.nt-1 ? A.n-n*A.nb : A.nb;
            // Plasma_ConjTrans will be converted to PlasmaTrans in
            // automatic datatype conversion, which is what we
            // want here.
            // PlasmaConjTrans is protected from this conversion.
            core_omp_zunmqr(
                PlasmaLeft, Plasma_ConjTrans,
                tempkm, tempnn, tempkm, ib, T.nb,
                A(k, k), ldak,
                T(k, k), T.mb,
                A(k, n), ldak,
                work,
                sequence, request);
        }
        for (m = k+1; m < A.mt; m++) {
            tempmm = m == A.mt-1 ? A.m-m*A.mb : A.mb;
            ldam = BLKLDD(A, m);
            core_omp_ztsqrt(
                tempmm, tempkn, ib, T.nb,
                A(k, k), ldak,
                A(m, k), ldam,
                T(m, k), T.mb,
                work,
                sequence, request);

            for (n = k+1; n < A.nt; n++) {
                tempnn = n == A.nt-1 ? A.n-n*A.nb : A.nb;
                core_omp_ztsmqr(
                    PlasmaLeft, Plasma_ConjTrans,
                    A.mb, tempnn, tempmm, tempnn, A.nb, ib, T.nb,
                    A(k, n), ldak,
                    A(m, n), ldam,
                    A(m, k), ldam,
                    T(m, k), T.mb,
                    work,
                    sequence, request);
            }
        }
    }
}
