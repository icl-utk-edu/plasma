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
#include <plasma_core_blas_zc.h>


#define As(m, n) (plasma_complex32_t*)plasma_tile_addr(As, m, n)
#define  A(m, n) (plasma_complex64_t*)plasma_tile_addr( A, m, n)

/***************************************************************************//**
 * Parallel tile conversion of matrix precision from single complex to
 * double complex.
 * @see plasma_omp_clag2z
 ******************************************************************************/
void plasma_pclag2z(plasma_desc_t As, plasma_desc_t A,
                    plasma_sequence_t *sequence, plasma_request_t *request)
{
    // Return if failed sequence.
    if (sequence->status != PlasmaSuccess)
        return;

    if (A.type == PlasmaGeneral && As.type == PlasmaGeneral) {
        for (int m = 0; m < As.mt; m++) {
            int am  = plasma_tile_mview(As, m);
            int lda = plasma_tile_mmain(As, m);
            int ldb = plasma_tile_mmain(A,  m);
            for (int n = 0; n < As.nt; n++) {
                int an = plasma_tile_nview(As, n);
                plasma_core_omp_clag2z(
                    am, an,
                    As(m, n), lda,
                    A(m, n),  ldb,
                    sequence, request);
            }
        }
    }
    else if (A.type == PlasmaGeneralBand &&
             As.type == PlasmaGeneralBand) {
        for (int n = 0; n < A.nt; n++ ) {
            int nvan = plasma_tile_nview(A, n);
            int m_start = (imax(0, n*A.nb-A.ku)) / A.nb;
            int m_end = (imin(A.m-1, (n+1)*A.nb+A.kl-1)) / A.nb;
            for (int m = m_start; m <= m_end; m++) {
                int ldam = plasma_tile_mmain_band(A, m, n);
                int mvam = plasma_tile_mview(A, m);
                plasma_core_omp_clag2z(
                    mvam, nvan,
                    As(m, n), ldam,
                    A(m, n), ldam,
                    sequence, request);
            }
        }
    }
}
