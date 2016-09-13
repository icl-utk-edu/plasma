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
#include "plasma_types.h"
#include "plasma_internal.h"
#include "core_blas_z.h"

#define tileA(m, n) ((plasma_complex64_t*)plasma_getaddr_band(uplo, A, (m), (n)))
#define bandA(m, n) (&(Af77[lda*(A.nb*(n)) + (uplo == PlasmaUpper ? A.ku : 0)+A.mb*((m)-(n))]))

/******************************************************************************/
void plasma_pzoocm2ccrb_band(plasma_enum_t uplo,
                             plasma_complex64_t *Af77, int lda, plasma_desc_t A,
                             plasma_sequence_t *sequence, plasma_request_t *request)
{
    int n, m;

    if (sequence->status != PLASMA_SUCCESS)
        return;

    for (n = 0; n < A.nt; n++)
    {
        int m_start, m_end;
        if (uplo == PlasmaFull) {
            m_start = (imax(0, n*A.nb-A.ku-A.kl)) / A.nb;
            m_end = (imin(A.m-1, (n+1)*A.nb+A.kl-1)) / A.nb;
        }
        else if (uplo == PlasmaUpper) {
            m_start = (imax(0, n*A.nb-A.ku)) / A.nb;
            m_end = (imin(A.m-1, (n+1)*A.nb-1)) / A.nb;
        }
        else {
            m_start = (imax(0, n*A.nb)) / A.nb;
            m_end = (imin(A.m-1, (n+1)*A.nb+A.kl-1)) / A.nb;
        }
        for (m = m_start; m <= m_end; m++)
        {
            int mb = imin(A.mb, A.m-m*A.mb);
            int nb = imin(A.nb, A.n-n*A.nb);
            core_omp_zlacpy_lapack2tile_band(
                   uplo, m, n,
                   mb, nb, A.mb, A.kl, A.ku,
                   bandA(m, n), lda-1,
                   tileA(m, n), BLKLDD_BAND(uplo, A, m, n));
                   //tileA(i_start,n), nb*nb, INOUT | GATHERV);
        }
    }
}
