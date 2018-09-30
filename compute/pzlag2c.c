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
#include <plasma_core_blas_zc.h>

/******************************************************************************/
static inline int imin(int a, int b)
{
    if (a < b)
        return a;
    else
        return b;
}

/******************************************************************************/
static inline int imax(int a, int b)
{
    if (a > b)
        return a;
    else
        return b;
}

#define  A(m, n) (plasma_complex64_t*)plasma_tile_addr( A, m, n)
#define As(m, n) (plasma_complex32_t*)plasma_tile_addr(As, m, n)

/***************************************************************************//**
 * Parallel tile conversion of matrix precision from double complex to
 * single complex.
 * @see plasma_omp_zlag2c
 *
 * If A and As are general band matrix they must have the same specs.
 ******************************************************************************/
void plasma_pzlag2c(plasma_desc_t A, plasma_desc_t As,
                    plasma_sequence_t *sequence, plasma_request_t *request)
{
    // Return if failed sequence.
    if (sequence->status != PlasmaSuccess)
        return;
    if (A.type == PlasmaGeneral && As.type == PlasmaGeneral) {
        for (int m = 0; m < A.mt; m++) {
            int am  = plasma_tile_mview(A,  m);
            int lda = plasma_tile_mmain(A,  m);
            int ldb = plasma_tile_mmain(As, m);
            for (int n = 0; n < A.nt; n++) {
                int an = plasma_tile_nview(A, n);
                plasma_core_omp_zlag2c(
                    am, an,
                    A(m, n),  lda,
                    As(m, n), ldb,
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
                plasma_core_omp_zlag2c(
                    mvam, nvan,
                    A(m, n), ldam,
                    As(m, n), ldam,
                    sequence, request);
            }
        }
    }
}
