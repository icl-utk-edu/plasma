/**
 *
 * @file
 *
 *  PLASMA is a software package provided by:
 *  University of Tennessee, US,
 *  University of Manchester, UK.
 *
 * @precisions mixed zc -> ds
 *
 **/

#include "plasma_async.h"
#include "plasma_descriptor.h"
#include "plasma_types.h"
#include "plasma_internal.h"
#include "core_blas_zc.h"

#define  A(m,n) ((plasma_complex64_t*) plasma_getaddr( A, m, n))
#define As(m,n) ((plasma_complex32_t*) plasma_getaddr(As, m, n))

/***************************************************************************//**
 * Parallel tile conversion of matrix precision from double complex to
 * single complex.
 * @see plasma_omp_zlag2c
 ******************************************************************************/
void plasma_pzlag2c(plasma_desc_t A, plasma_desc_t As,
                    plasma_sequence_t *sequence, plasma_request_t *request)
{
    int X, Y;
    int m, n;
    int ldam, ldbm;

    // Check sequence status
    if (sequence->status != PlasmaSuccess) {
        plasma_request_fail(sequence, request, PlasmaErrorSequence);
        return;
    }

    for (m = 0; m < A.mt; m++) {
        X = m == A.mt-1 ? A.m-m*A.mb : A.mb;
        ldam = BLKLDD(A, m);
        ldbm = BLKLDD(As, m);
        for (n = 0; n < A.nt; n++) {
            Y = n == A.nt-1 ? A.n-n*A.nb : A.nb;
            core_omp_zlag2c(
                X, Y,
                A(m, n),  ldam,
                As(m, n), ldbm);
        }
    }
}
