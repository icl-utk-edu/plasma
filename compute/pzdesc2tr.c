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
#include "plasma_internal.h"
#include "plasma_types.h"
#include "plasma_workspace.h"
#include <plasma_core_blas.h>

/******************************************************************************/
void plasma_pzdesc2tr(plasma_desc_t A,
                      plasma_complex64_t *pA, int lda,
                      plasma_sequence_t *sequence,
                      plasma_request_t *request)
{
    // Return if failed sequence.
    if (sequence->status != PlasmaSuccess)
        return;

    for (int m = 0; m < A.mt; m++) {
        int ldt = plasma_tile_mmain(A, m);
        int n_start = (A.type == PlasmaUpper ? m : 0);
        int n_end   = (A.type == PlasmaUpper ? A.nt : m+1);
        for (int n = n_start; n < n_end; n++) {
            int x1 = n == 0 ? A.j%A.nb : 0;
            int y1 = m == 0 ? A.i%A.mb : 0;
            int x2 = n == A.nt-1 ? (A.j+A.n-1)%A.nb+1 : A.nb;
            int y2 = m == A.mt-1 ? (A.i+A.m-1)%A.mb+1 : A.mb;

            plasma_complex64_t *f77 = &pA[(size_t)A.nb*lda*n + (size_t)A.mb*m];
            plasma_complex64_t *bdl =
                (plasma_complex64_t*)plasma_tile_addr(A, m, n);

            plasma_core_omp_zlacpy(PlasmaGeneral, PlasmaNoTrans,
                            y2-y1, x2-x1,
                            &(bdl[x1*A.nb+y1]), ldt,
                            &(f77[x1*lda+y1]), lda,
                            sequence, request);
        }
    }
}
