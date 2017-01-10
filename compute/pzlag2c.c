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
#include "plasma_internal_zc.h"
#include "core_blas_zc.h"

#define  A(m, n) (plasma_complex64_t*)plasma_tile_addr( A, m, n)
#define As(m, n) (plasma_complex32_t*)plasma_tile_addr(As, m, n)

/***************************************************************************//**
 * Parallel tile conversion of matrix precision from double complex to
 * single complex.
 * @see plasma_omp_zlag2c
 ******************************************************************************/
void plasma_pzlag2c(plasma_desc_t A, plasma_desc_t As,
                    plasma_sequence_t *sequence, plasma_request_t *request)
{
    // Return if failed sequence.
    if (sequence->status != PlasmaSuccess)
        return;

    for (int m = 0; m < A.mt; m++) {
        int am  = plasma_tile_mview(A,  m);
        int lda = plasma_tile_mmain(A,  m);
        int ldb = plasma_tile_mmain(As, m);
        for (int n = 0; n < A.nt; n++) {
            int an = plasma_tile_nview(A, n);
            core_omp_zlag2c(
                am, an,
                A(m, n),  lda,
                As(m, n), ldb,
                sequence, request);
        }
    }
}
