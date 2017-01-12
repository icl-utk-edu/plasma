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
#include "plasma_internal.h"
#include "plasma_types.h"
#include "core_blas.h"

#define A(m, n) (plasma_complex64_t*)plasma_tile_addr(A, m, n)

/******************************************************************************/
void plasma_pzlaswp(plasma_enum_t colrow,
                    plasma_desc_t A, int *ipiv, int incx,
                    plasma_sequence_t *sequence, plasma_request_t *request)
{
    // Return if failed sequence.
    if (sequence->status != PlasmaSuccess)
        return;

    if (colrow == PlasmaRowwise) {
        for (int n = 0; n < A.nt; n++) {
            plasma_complex64_t *a00, *a10;

            a00 = A(0, n);
            a10 = A(A.mt-1, n);

            int ma00 = (A.mt-1)*A.mb;
            int na00 = plasma_tile_nmain(A, n);

            int lda10 = plasma_tile_mmain(A, A.mt-1);
            int nva10 = plasma_tile_nview(A, n);

            #pragma omp task depend (inout:a00[ma00*na00]) \
                             depend (inout:a10[lda10*nva10])
            {
                int nvan = plasma_tile_nview(A, n);
                plasma_desc_t view = plasma_desc_view(A, 0, n*A.nb, A.m, nvan);
                core_zlaswp(colrow, view, 1, A.m, ipiv, incx);
            }
        }
    }
    else {
        for (int m = 0; m < A.mt; m++) {
            int mvam = plasma_tile_mview(A, m);
            plasma_desc_t view = plasma_desc_view(A, m*A.mb, 0, mvam, A.n);
            core_zlaswp(colrow, view, 1, A.n, ipiv, incx);
        }
    }
}
