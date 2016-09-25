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
 * Parallel tile conversion of matrix precision from single complex to
 * double complex.
 * @see plasma_omp_clag2z
 ******************************************************************************/
void plasma_pclag2z(plasma_desc_t As, plasma_desc_t A,
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

    for (m = 0; m < As.mt; m++) {
        X = m == As.mt-1 ? As.m-m*As.mb : As.mb;
        ldam = BLKLDD(As, m);
        ldbm = BLKLDD(A, m);
        for (n = 0; n < As.nt; n++) {
            Y = n == As.nt-1 ? As.n-n*As.nb : As.nb;
            core_omp_clag2z(
                X, Y,
                As(m, n), ldam,
                A(m, n),  ldbm);
        }
    }
}
