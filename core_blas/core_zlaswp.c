/**
 *
 * @file
 *
 *  PLASMA is a software package provided by:
 *  University of Tennessee, US,
 *  University of Manchester, UK.
 *
 * @precisions normal z -> c d s
 *
 **/

#include "core_blas.h"
#include "plasma_internal.h"
#include "plasma_types.h"
#include "core_lapack.h"

#define A(m, n) (plasma_complex64_t*)plasma_tile_addr(A, m, n)

/******************************************************************************/
void core_zlaswp(plasma_enum_t colrow,
                 plasma_desc_t A, int k1, int k2, int *ipiv, int incx)
{
    //=================
    // PlasmaRowwise
    //=================
    if (colrow == PlasmaRowwise) {
        if (incx > 0) {
            for (int m = k1-1; m <= k2-1; m += incx) {
                if (ipiv[m]-1 != m) {

                    int m1 = m;
                    int m2 = ipiv[m]-1;

                    int lda1 = plasma_tile_mmain(A, m1/A.mb);
                    int lda2 = plasma_tile_mmain(A, m2/A.mb);

                    cblas_zswap(A.n,
                                A(m1/A.mb, 0) + m1%A.mb, lda1,
                                A(m2/A.mb, 0) + m2%A.mb, lda2);
                }
            }
        }
        else {
            for (int m = k2-1; m >= k1-1; m += incx) {
                if (ipiv[m]-1 != m) {

                    int m1 = m;
                    int m2 = ipiv[m]-1;

                    int lda1 = plasma_tile_mmain(A, m1/A.mb);
                    int lda2 = plasma_tile_mmain(A, m2/A.mb);

                    cblas_zswap(A.n,
                                A(m1/A.mb, 0) + m1%A.mb, lda1,
                                A(m2/A.mb, 0) + m2%A.mb, lda2);
                }
            }        
        }
    }
    //=================
    // PlasmaColwise
    //=================
    else {

    }
}
