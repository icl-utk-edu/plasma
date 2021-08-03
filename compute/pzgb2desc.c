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
void plasma_pzgb2desc(plasma_complex64_t *pA, int lda,
                      plasma_desc_t A,
                      plasma_sequence_t *sequence,
                      plasma_request_t *request)
{
    // Return if failed sequence.
    if (sequence->status != PlasmaSuccess)
        return;

    plasma_complex64_t *f77;
    plasma_complex64_t *bdl;

    int x1, y1;
    int x2, y2;
    int n, m, ldt;
    for (m = 0; m < A.mt; m++) {
        for (n = 0; n < A.nt; n++) {
            // don't want to copy tiles without elements (plasma_tile_addr)
            // cannot handle it.
            // Calculate kut without normal space for transformations.
            if(m-n >= A.klt || n-m >= 1+(A.ku+A.nb-1)/A.nb)
            {
                //// if you notice any bugs, in the copying of matrices,
                //// check back on this conditional
                continue;
            }
            ldt = plasma_tile_mmain_band(A, m, n); // possibly too many calls.
                                    // plasma_tile_mmain_band need not know m,n
            x1 = n == 0 ? A.j%A.nb : 0;
            y1 = m == 0 ? A.i%A.mb : 0;
            x2 = n == A.nt-1 ? (A.j+A.n-1)%A.nb+1 : A.nb;
            y2 = m == A.mt-1 ? (A.i+A.m-1)%A.mb+1 : A.mb;


            f77 = &pA[(size_t)A.nb*lda*n + (size_t)A.mb*m];
            bdl = (plasma_complex64_t*)plasma_tile_addr(A, m, n);
            // printf("[%s] Full view:\nm=%d\nn=%d\nx1=%d\ny1=%d\nx2=%d\ny2=%d\nlda=%d\n",
                       // __FILE__,m,n,x1,y1,x2,y2,lda);
            // printf("pA(%d,%d) = %1.3f (%u)\n", m, n,
                   // pA[(size_t)A.nb*lda*n + (size_t)A.mb*m],
                   // &pA[(size_t)A.nb*lda*n + (size_t)A.mb*m]-
                   // &pA[(size_t)A.nb*lda*0 + (size_t)A.mb*0]);
            plasma_core_omp_zlacpy(
                            PlasmaGeneralBand, PlasmaNoTrans,
                            y2-y1, x2-x1,
                            &(f77[x1*lda+y1]), lda,
                            &(bdl[x1*A.nb+y1]), ldt,
                            sequence, request);
        }
    }
}
